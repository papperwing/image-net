package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.dataset.processor.DataSetTransform;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.EarlyStoppingParallelTrainer;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.deeplearning4j.ui.weights.ConvolutionalIterationListener;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Trainer for training neural network.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNetTrainer {

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    private final double splitPercentage = 0.8; //TODO: replace with Configuration setting

    protected final Configuration conf;
    protected final NeuralNetModelWrapper modelWrapper;

    /**
     * Constructor of {@link ImageNetTrainer}
     *
     * @param conf global configuration
     */
    public ImageNetTrainer(
            final Configuration conf,
            final NeuralNetModelWrapper modelWrapper
    ) {
        this.conf = conf;
        this.modelWrapper = modelWrapper;
        logger.info("Memory usage for javacpp:\n" +
                "maxbytes: " + Pointer.maxBytes() + "\n" +
                "maxPhysicalBytes: " + Pointer.maxPhysicalBytes());
    }

    /**
     * @param dataset
     * @return
     */
    public NeuralNetModelWrapper trainModel(
            final DataSet dataset
    ) {

        Nd4j.getMemoryManager().setAutoGcWindow(2500);

        if (this.conf.isDebug()) {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
        }

        final DataSet testSet = dataset.split(splitPercentage);

        /*get statistics of datasets after split*/
        printDatasetStatistics(dataset);
        printDatasetStatistics(testSet);

        final DataSetTransform processor = new DataSetTransform(conf, modelWrapper.getType());

        final DataSetIterator trainIterator = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(dataset),
                modelWrapper.getType(),
                "train"
        );

        final DataSetIterator testIterator = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(testSet),
                modelWrapper.getType(),
                "test"
        );

        return trainModel(
                testIterator,
                trainIterator,
                dataset.getLabels()
        );
    }

    /**
     * @param testIterator
     * @param trainIterator
     * @param labels
     * @return
     */
    protected NeuralNetModelWrapper trainModel(
            DataSetIterator testIterator,
            DataSetIterator trainIterator,
            List<Label> labels
    ) {
        setupStatInterface(modelWrapper.getModel());

        final EarlyStoppingResult<Model> result;
        if (this.conf.getGPUCount() == 1) {
            result = runEarlyStopping(
                    modelWrapper.getModel(),
                    trainIterator,
                    testIterator,
                    this.conf.getTempFolder() + File.separator + "model"
            );
        } else {
            CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
            result = runEarlyStoppingGPU(
                    modelWrapper.getModel(),
                    trainIterator,
                    testIterator,
                    this.conf.getTempFolder() + File.separator + "model"
            );
        }

        logger.info("Termination reason: " + result.getTerminationReason());
        logger.info("Termination details: " + result.getTerminationDetails());
        logger.info("Total epochs: " + result.getTotalEpochs());
        logger.info("Best epoch number: " + result.getBestModelEpoch());
        logger.info("Score at best epoch: " + result.getBestModelScore());

        return new NeuralNetModelWrapper(result.getBestModel(), labels, ModelType.ALEXNET);

    }

    /**
     * Adding {@link IterationListener} into modelWrapper for getting further information about process of learning.
     * <b>It is necessary for correct behaviour to have logger backend for SL4J correctly set.</b>
     *
     * @param model {@link Model} into which are {@link IterationListener} set
     */
    private void setupStatInterface(
            Model model
    ) {
        if(this.conf.getUIMode() != null) {
            List<IterationListener> iterationListeners = Arrays.asList(uiSetting());
            iterationListeners.add(new ScoreIterationListener(1));
            model.setListeners(iterationListeners.toArray(new IterationListener[0]));
        }
        else{
            model.setListeners(new ScoreIterationListener(1));
        }
    }

    private IterationListener[] uiSetting(){
        //TODO: implement REMOTE and set STORAGE default
        switch (this.conf.getUIMode()){
            case ONLINE:
                if(this.conf.getJavaMinorVersion() < 8) throw new UnsupportedOperationException("Online cannot be used in Java 7");
                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage= new InMemoryStatsStorage();
                uiServer.attach(statsStorage);
                return new IterationListener[]{
                        new StatsListener(statsStorage),
                        new ConvolutionalIterationListener(
                                statsStorage,
                                10,
                                false)};
            case REMOTE:
                throw  new UnsupportedOperationException("Not implemented yet.");
            case STORAGE:
                File storeFile = new File(this.conf.getTempFolder() + "/model/storage_file");
                IterationListener statListener;
                if (this.conf.getJavaMinorVersion() == 7) {
                    statListener = new J7StatsListener(
                            new J7FileStatsStorage(
                                    storeFile
                            )
                    );
                } else if (this.conf.getJavaMinorVersion() > 7) {
                    statListener = new StatsListener(
                            new FileStatsStorage(
                                    storeFile
                            )
                    );
                } else {
                    throw new IllegalStateException("version of java is too small. Upgrade your Java at least to 1.7");
                }
                return new IterationListener[]{statListener};
        }
        throw new IllegalStateException("One of previous must be selected");
    }


    /**
     * Method print information about dataset label distribution
     *
     * @param set {@link DataSet} containing data samples
     */
    private void printDatasetStatistics(
            final DataSet set
    ) {
        final Map<Label, Integer> labelDistribution = set.getLabelDistribution();

        final StringBuilder statistic = new StringBuilder(System.lineSeparator());
        int maxNameLength = 0;
        int maxValueLenght = 0;
        for (final Label label : labelDistribution.keySet()) {
            maxNameLength = Math.max(label.getLabelName().length(), maxNameLength);
            maxValueLenght = Math.max(String.valueOf(labelDistribution.get(label)).length(), maxValueLenght);
        }

        String pattern = "%-" + maxNameLength + 5 + "s%-" + maxValueLenght + "d";

        for (final Label label : labelDistribution.keySet()) {
            statistic.append(
                    String.format(
                            pattern,
                            label.getLabelName(),
                            labelDistribution.get(label)
                    )
            );
            statistic.append(System.lineSeparator());
        }

        logger.info(statistic.toString());
    }


    /**
     * @param model
     * @param trainDataSet
     * @param testDataSet
     * @param tempDirLoc
     * @return
     */
    private EarlyStoppingResult runEarlyStopping(
            final Model model,
            final DataSetIterator trainDataSet,
            final DataSetIterator testDataSet,
            final String tempDirLoc
    ) {

        EarlyStoppingResult result = null;

        if (model instanceof MultiLayerNetwork) {

            final EarlyStoppingConfiguration.Builder<MultiLayerNetwork> builder = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(this.conf.getEpoch()))
                    .modelSaver(new LocalFileModelSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculator(testDataSet, true))
                    .evaluateEveryNEpochs(1);

            if (this.conf.isTimed()) {
                builder.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(this.conf.getTime(), TimeUnit.MINUTES));
            }
            final EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = builder.build();
            final IEarlyStoppingTrainer<MultiLayerNetwork> trainer
                    = new EarlyStoppingTrainer(
                    esConfig,
                    (MultiLayerNetwork) model,
                    trainDataSet,
                    null
            );
            result = trainer.fit();
        } else if (model instanceof ComputationGraph) {

            final EarlyStoppingConfiguration.Builder<ComputationGraph> builder = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(this.conf.getEpoch()))
                    .modelSaver(new LocalFileGraphSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculatorCG(testDataSet, true))
                    .evaluateEveryNEpochs(1);

            if (this.conf.isTimed()) {
                builder.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(this.conf.getTime(), TimeUnit.MINUTES));
            }
            final EarlyStoppingConfiguration<ComputationGraph> esConfig = builder.build();
            final IEarlyStoppingTrainer<ComputationGraph> trainer
                    = new EarlyStoppingGraphTrainer(
                    esConfig,
                    (ComputationGraph) model,
                    trainDataSet,
                    null
            );
            result = trainer.fit();
        }

        return result;
    }

    /**
     * @param model
     * @param trainDataSet
     * @param testDataSet
     * @param tempDirLoc
     * @return
     */
    private EarlyStoppingResult runEarlyStoppingGPU(
            final Model model,
            final DataSetIterator trainDataSet,
            final DataSetIterator testDataSet,
            final String tempDirLoc
    ) {

        EarlyStoppingResult result = null;

        if (model instanceof MultiLayerNetwork) {

            final EarlyStoppingConfiguration.Builder<MultiLayerNetwork> builder = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(this.conf.getEpoch()))
                    .modelSaver(new LocalFileModelSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculator(testDataSet, true))
                    .evaluateEveryNEpochs(1);

            if (this.conf.isTimed()) {
                builder.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(this.conf.getTime(), TimeUnit.MINUTES));
            }
            final EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = builder.build();
            final IEarlyStoppingTrainer<MultiLayerNetwork> trainer
                    = new EarlyStoppingParallelTrainer<>(
                    esConfig,
                    (MultiLayerNetwork) model,
                    trainDataSet,
                    null,
                    this.conf.getGPUCount(),
                    1,
                    1
            );
            result = trainer.fit();
        } else if (model instanceof ComputationGraph) {

            final EarlyStoppingConfiguration.Builder<ComputationGraph> builder = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(this.conf.getEpoch()))
                    .modelSaver(new LocalFileGraphSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculatorCG(testDataSet, true))
                    .evaluateEveryNEpochs(1);

            if (this.conf.isTimed()) {
                builder.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(this.conf.getTime(), TimeUnit.MINUTES));
            }
            final EarlyStoppingConfiguration<ComputationGraph> esConfig = builder.build();
            final IEarlyStoppingTrainer<ComputationGraph> trainer
                    = new EarlyStoppingParallelTrainer<>(
                    esConfig,
                    (ComputationGraph) model,
                    trainDataSet,
                    null,
                    this.conf.getGPUCount(),
                    1,
                    1
            );
            result = trainer.fit();
        }

        return result;
    }

    private Model testRunEarlyStoppingGPU(
            final Model model,
            final DataSetIterator trainDataSet,
            final DataSetIterator testDataSet,
            final String tempDirLoc
    ) {
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .prefetchBuffer(1)
                .workers(this.conf.getGPUCount())
                .averagingFrequency(1)
                .reportScoreAfterAveraging(true)
                .workspaceMode(WorkspaceMode.SEPARATE)
                .build();

        int actualEpoch = 0;
        while (actualEpoch < this.conf.getEpoch()) {
            wrapper.fit(trainDataSet);
        }
        return model;
    }
}
