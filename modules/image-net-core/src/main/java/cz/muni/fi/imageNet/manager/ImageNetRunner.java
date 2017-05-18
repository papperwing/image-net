package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import cz.muni.fi.imageNet.Pojo.ModelType;
import cz.muni.fi.imageNet.Pojo.NeuralNetModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    //TODO: odstranit a narvat do konfigurace
    private final int height = 224;
    private final int width = 224;
    private final int channels = 3;

    private final int batchSize = 5;
    private final double treshold = 0.5;
    private final double splitPercentage = 0.8;

    private final Configuration conf;

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
     * @param startTime
     * @return
     */
    public NeuralNetModel trainModel(final NeuralNetModel model, DataSet dataset, long startTime) {

        DataSet testSet = dataset.split(splitPercentage);

        /*get statistics of datasets after split*/
        printDatasetStatistics(dataset);
        printDatasetStatistics(testSet);

        final DataSetIterator trainIterator = prepareDataSetIterator(dataset, model.getType());

        final DataSetIterator testIterator = prepareDataSetIterator(testSet, model.getType());

        Nd4j.getMemoryManager().setAutoGcWindow(2500);

        EarlyStoppingResult<ComputationGraph> result = runEarlyStoppingTrain(
                model.getModel(),
                trainIterator,
                testIterator,
                this.conf.getTempFolder()
        );

        return new NeuralNetModel(result.getBestModel(), dataset.getLabels(), ModelType.VGG16);
    }

    /**
     *
     *
     * @param model
     * @param imageLocation
     * @return
     */
    public List<List<Label>> classify(final NeuralNetModel model, String[] imageLocations) {

        try {
            List<INDArray> images = new ArrayList<>();
            for (String imageLocation : imageLocations) {
                images.add(generateINDArray(new File(imageLocation)));
            }

            INDArray[] imageFeatures = images.toArray(new INDArray[images.size()]);
            return getLabel(
                    new ArrayList(model.getLabels()),
                    model.getModel().output(imageFeatures)
            );
        } catch (IOException ex) {
            logger.error("Loading of image was not sucessfull.", ex);
        }
        return null;
    }

    private DataSetIterator prepareDataSetIterator(DataSet dataset, ModelType modelType) {
        //it is necessarry to use ImageNetSplit to stay consist, because ImageNetRecordReader force to use ImageNetSplit
        INDASerializer.conf = this.conf;
        ImageNetSplit is = new ImageNetSplit(dataset);
        ImageNetRecordReader recordReader = new ImageNetRecordReader(
                height,
                width,
                channels
        );
        try {
            logger.debug("Record reader inicialization.");
            recordReader.initialize(is);
            DataSetIterator dataIter = new RecordReaderDataSetIterator(
                    recordReader,
                    batchSize
            );
            switch (modelType) {
                case VGG16:
                    dataIter.setPreProcessor(TrainedModels.VGG16.getPreProcessor());
                    break;
            }

            return new AsyncDataSetIterator(dataIter, 3, true);
        } catch (IOException ex) {
            logger.error("Loading of image was not sucessfull.", ex);
        } catch (InterruptedException ex) {
            logger.error("Transformation of file into URI wasnot sucessfull.", ex);
        }

        return null;
    }

    private List<List<Label>> getLabel(List<Label> labels, INDArray... outputs) {
        //TODO: Vylepšit na iterování skrze INDArray
        List<List<Label>> results = new ArrayList();
        for (INDArray output : outputs) {
            List<Label> result = new ArrayList();
            double[] asDouble = output.dup().data().asDouble();
            for (int i = 0; i < asDouble.length; i++) {
                if (asDouble[i] > treshold) {
                    result.add(labels.get(i));
                }
            }
            results.add(result);
        }
        return results;
    }

    private INDArray generateINDArray(File image) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray imageVector = loader.asMatrix(image);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageVector);
        return imageVector;
    }

    private EarlyStoppingResult runEarlyStoppingTrain(
            ComputationGraph model,
            DataSetIterator trainDataSet,
            DataSetIterator testDataSet,
            String tempDirLoc
    ) {
        EarlyStoppingConfiguration.Builder<ComputationGraph> builder = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(this.conf.getEpoch()))
                .modelSaver(new LocalFileGraphSaver(tempDirLoc))
                .scoreCalculator(new DataSetLossCalculatorCG(testDataSet, true));

        if (this.conf.isTimed()) {
            builder.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(this.conf.getTime(), TimeUnit.MINUTES));
        }
        EarlyStoppingConfiguration<ComputationGraph> esConfig = builder.build();
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConfig, model, trainDataSet);

        return trainer.fit();
    }

    private void printDatasetStatistics(DataSet set) {
        Map<Label, Integer> labelDistribution = set.getLabelDistribution();

        StringBuilder statistic = new StringBuilder(System.lineSeparator());
        int maxNameLength = 0;
        int maxValueLenght = 0;
        for (Label label : labelDistribution.keySet()) {
            maxNameLength = Math.max(label.getLabelName().length(), maxNameLength);
            maxValueLenght = Math.max(String.valueOf(labelDistribution.get(label)).length(), maxValueLenght);
        }

        String pattern = "%-" + maxNameLength + 5 + "s%-" + maxValueLenght + "d";

        for (Label label : labelDistribution.keySet()) {
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

}
