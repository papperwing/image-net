package cz.muni.fi.imageNet.core.manager;

import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.Label;
import cz.muni.fi.imageNet.core.objects.ModelType;
import cz.muni.fi.imageNet.core.objects.NeuralNetModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.datavec.api.berkeley.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RandomCropTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Class is used for running neural network base action:
 * <ul><li>training model</li><li>clasification</li></ul>
 *
 * @author Jakub Peschel
 */
public class ImageNetRunner {

    private final Logger logger = LoggerFactory.getLogger(ImageNetRunner.class);

    private final int height = 224;
    private final int width = 224;
    private final int channels = 3;

    private final int batchSize = 32;
    private final double treshold = 0.5;
    private final double splitPercentage = 0.8;

    protected final Configuration conf;

    /**
     *
     * @param conf
     */
    public ImageNetRunner(Configuration conf) {
        this.conf = conf;
    }

    /**
     *
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

        final DataSetIterator trainIterator = prepareDataSetIterator(dataset, model.getType(), "train");

        final DataSetIterator testIterator = prepareDataSetIterator(testSet, model.getType(), "test");

        System.gc();//To ensure memory clearing
        setupStatInterface(model.getModel());
        final EarlyStoppingResult<Model> result = runEarlyStoppingTrain(
                model.getModel(),
                trainIterator,
                testIterator,
                this.conf.getTempFolder() + File.separator + "model"
        );

        return new NeuralNetModel(result.getBestModel(), dataset.getLabels(), ModelType.VGG16);
    }

    /**
     *
     *
     * @param modelWrapper
     * @param imageLocations
     * @return
     */
    public List<List<Label>> classify(
            final NeuralNetModel modelWrapper,
            final String[] imageLocations
    ) {

        try {
            final List<INDArray> images = new ArrayList<>();
            for (final String imageLocation : imageLocations) {
                images.add(generateINDArray(new File(imageLocation)));
            }

            Model model = modelWrapper.getModel();
            List<INDArray> outputArray = new ArrayList<>();
            for (INDArray input : images) {
                if (model instanceof MultiLayerNetwork) {
                    outputArray.add(((MultiLayerNetwork) model).output(input));
                } else {
                    outputArray.add(((ComputationGraph) model).outputSingle(input));

                }
            }
            INDArray[] iNDArrays = new INDArray[outputArray.size()];
            iNDArrays = outputArray.toArray(iNDArrays);
            return getLabel(
                    new ArrayList(modelWrapper.getLabels()),
                    iNDArrays
            );
        } catch (IOException ex) {
            logger.error("Loading of image was not sucessfull.", ex);
        }
        return null;
    }

    protected DataSetIterator prepareDataSetIterator(
            final DataSet dataset,
            final ModelType modelType,
            final String saveDataName
    ) {
        final List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(new Pair(new ResizeImageTransform(300, 300), 1.0));
        pipeline.add(new Pair(new RandomCropTransform(224, 224), 1.0));
        pipeline.add(new Pair(new FlipImageTransform(1), 0.5));
        final ImageTransform combinedTransform
                = new MultiImageTransform(
                        new ImageTransform[]{
                            new PipelineImageTransform(pipeline, false)
                        }
                );

        final ImageNetRecordReader recordReader = new ImageNetRecordReader(
                height,
                width,
                channels,
                combinedTransform
        );
        logger.debug("Record reader inicialization.");
        recordReader.initialize(dataset, new ImagePreProcessingScaler());
        final DataSetIterator dataIter
                = new AsyncDataSetIterator(
                        new RecordReaderDataSetIterator(
                                recordReader,
                                batchSize,
                                1,
                                dataset.getLabels().size(),
                                true
                        )
                );
        logger.debug("PreSaving dataset for faster processing");
        final File saveFolder = new File(this.conf.getTempFolder() + File.separator + "minibatches" + File.separator + saveDataName);
        saveFolder.mkdirs();
        saveFolder.deleteOnExit();
        int dataSaved = 0;
        while (dataIter.hasNext()) {
            final org.nd4j.linalg.dataset.DataSet next = dataIter.next();

            logger.debug("" + dataSaved);
            next.save(new File(saveFolder, saveDataName + "-" + dataSaved + ".bin"));
            dataSaved++;
        }
        logger.debug("DataSet presaved");
        return new ExistingMiniBatchDataSetIterator(saveFolder, saveDataName + "-%d.bin");
    }

    private List<List<Label>> getLabel(
            final List<Label> labels,
            final INDArray... outputs
    ) {
        //TODO: Vylepšit na iterování skrze INDArray
        final List<List<Label>> results = new ArrayList();
        for (INDArray output : outputs) {
            final List<Label> result = new ArrayList();
            final double[] asDouble = output.dup().data().asDouble();
            for (int i = 0; i < asDouble.length; i++) {
                if (asDouble[i] > treshold) {
                    result.add(labels.get(i));
                }
            }
            results.add(result);
        }
        return results;
    }

    private INDArray generateINDArray(
            final File image
    ) throws IOException {
        final NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        final INDArray imageVector = loader.asMatrix(image);
        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageVector);
        return imageVector;
    }

    private EarlyStoppingResult runEarlyStoppingTrain(
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
            final EarlyStoppingTrainer trainer
                    = new EarlyStoppingTrainer(
                            esConfig,
                            (MultiLayerNetwork) model,
                            trainDataSet);
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
            final EarlyStoppingGraphTrainer trainer
                    = new EarlyStoppingGraphTrainer(
                            esConfig,
                            (ComputationGraph) model,
                            trainDataSet);
            result = trainer.fit();
        }

        logger.info("Termination reason: " + result.getTerminationReason());
        logger.info("Termination details: " + result.getTerminationDetails());
        logger.info("Total epochs: " + result.getTotalEpochs());
        logger.info("Best epoch number: " + result.getBestModelEpoch());
        logger.info("Score at best epoch: " + result.getBestModelScore());

        return result;
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

    private void setupStatInterface(
            Model model
    ) {

        StatsStorage store = new J7FileStatsStorage(new File(this.conf.getImageDownloadFolder() + "/../storage_file"));

        model.setListeners(new J7StatsListener(store), new ScoreIterationListener(1));

    }

    public String evaluateModel(
            NeuralNetModel modelWrapper, DataSet set
    ) {
        DataSetIterator iter = prepareDataSetIterator(set, ModelType.LENET, "evaluation");

        final EvaluationBinary evaluationBinary = new EvaluationBinary(set.getLabels().size(), null);
        Model model = modelWrapper.getModel();
        if (model instanceof MultiLayerNetwork) {
            ((MultiLayerNetwork) model).doEvaluation(iter, evaluationBinary);
        } else if (model instanceof ComputationGraph) {
            ((ComputationGraph) model).doEvaluation(iter, evaluationBinary);
        }
        return evaluationBinary.stats();
    }

}
