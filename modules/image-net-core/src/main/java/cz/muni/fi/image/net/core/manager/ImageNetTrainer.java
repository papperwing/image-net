package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.dataset.processor.DataSetProcessor;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.EarlyStoppingParallelTrainer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Trainer for training neural network.
 *
 * @author Jakub Peschel
 */
public class ImageNetTrainer {

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    private final double splitPercentage = 0.8;

    protected final Configuration conf;

    /**
     * @param conf
     */
    public ImageNetTrainer(Configuration conf) {
        this.conf = conf;
    }

    /**
     * @param model
     * @param dataset
     * @return
     */
    public NeuralNetModel trainModel(
            final NeuralNetModel model,
            final DataSet dataset
    ) {

        Nd4j.getMemoryManager().setAutoGcWindow(2500);

        final DataSet testSet = dataset.split(splitPercentage);

        /*get statistics of datasets after split*/
        printDatasetStatistics(dataset);
        printDatasetStatistics(testSet);

        DataSetProcessor processor = new DataSetProcessor(conf, model.getType());

        final DataSetIterator trainIterator = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(dataset),
                model.getType(),
                "train"
        );

        final DataSetIterator testIterator = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(testSet),
                model.getType(),
                "test"
        );

        setupStatInterface(model.getModel());

        final EarlyStoppingResult<Model> result;
        if (this.conf.getGPUCount() == 1) {
            result = runEarlyStopping(
                    model.getModel(),
                    trainIterator,
                    testIterator,
                    this.conf.getTempFolder() + File.separator + "model"
            );
        } else {
            result = runEarlyStoppingGPU(
                    model.getModel(),
                    trainIterator,
                    testIterator,
                    this.conf.getTempFolder() + File.separator + "model"
            );
        }

        return new NeuralNetModel(result.getBestModel(), dataset.getLabels(), ModelType.VGG16);
    }

    private void setupStatInterface(
            Model model
    ) {

        StatsStorage store = new J7FileStatsStorage(new File(this.conf.getImageDownloadFolder() + "/../storage_file"));

        model.setListeners(new J7StatsListener(store), new ScoreIterationListener(1));
    }

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
                    .modelSaver(new LocalFileGraphSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculatorCG(testDataSet, true))
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

        logger.info("Termination reason: " + result.getTerminationReason());
        logger.info("Termination details: " + result.getTerminationDetails());
        logger.info("Total epochs: " + result.getTotalEpochs());
        logger.info("Best epoch number: " + result.getBestModelEpoch());
        logger.info("Score at best epoch: " + result.getBestModelScore());

        return result;
    }

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
                    .modelSaver(new LocalFileGraphSaver(tempDirLoc))
                    .scoreCalculator(new DataSetLossCalculatorCG(testDataSet, true))
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
                    1,
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

        logger.info("Termination reason: " + result.getTerminationReason());
        logger.info("Termination details: " + result.getTerminationDetails());
        logger.info("Total epochs: " + result.getTotalEpochs());
        logger.info("Best epoch number: " + result.getBestModelEpoch());
        logger.info("Score at best epoch: " + result.getBestModelScore());

        return result;
    }
}
