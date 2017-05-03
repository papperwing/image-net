package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import cz.muni.fi.imageNet.Pojo.ModelType;
import cz.muni.fi.imageNet.Pojo.NeuralNetModel;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
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
        
        DataSet testSet = dataset.split(0.8);
        final DataSetIterator trainIterator = prepareDataSetIterator(dataset, model.getType());
        final DataSetIterator testIterator = prepareDataSetIterator(testSet, model.getType());

        for (int n = 0; n < conf.getEpoch(); n++) {

            EvaluationBinary eval = new EvaluationBinary(dataset.getLabelStrings().size());
            eval.setLabelNames(dataset.getLabelStrings());
            logger.debug("Starting to train in " + n + " epoch");
            Nd4j.getMemoryManager().setAutoGcWindow(2500);
            model.getModel().fit(trainIterator);
            logger.debug("Starting to test after " + n + " epoch");
            eval.merge(customEval(model.getModel(), dataset.getLabelStrings(), testIterator));
            logger.info("[epoch:" + n + "]:\n" + eval.stats());
            trainIterator.reset();
            testIterator.reset();
            if (this.conf.isTimed()
                    && startTime + this.conf.getTime() >= System.currentTimeMillis()) {
                return model;

            }
        }
        return model;
    }

    public EvaluationBinary customEval(
            ComputationGraph model,
            List<String> labelsList,
            DataSetIterator datasetIterator
    ) {
        EvaluationBinary eval = new EvaluationBinary(labelsList.size());

        return model.doEvaluation(datasetIterator, eval);
    }

    /**
     *
     *
     * @param model
     * @param imageLocation
     * @return
     */
    public List<Label> clasify(final NeuralNetModel model, String imageLocation) {

        try {
            model.getModel().init();

            File image = new File(imageLocation);
            INDArray imageFeatures = generateINDArray(image);
            return getLabel(
                    new ArrayList(model.getLabels()),
                    model.getModel().outputSingle(imageFeatures)
            );
        } catch (IOException ex) {
            logger.error("Loading of image was not sucessfull.", ex);
        }
        return null;
    }

    private DataSetIterator prepareDataSetIterator(DataSet dataset, ModelType modelType) {
        //it is necessarry to use ImageNetSplit to stay consist, because ImageNetRecordReader force to use ImageNetSplit
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

    private List<Label> getLabel(List<Label> labels, INDArray output) {
        //TODO: Vylepšit na iterování skrze INDArray
        List<Label> result = new ArrayList();
        double[] asDouble = output.dup().data().asDouble();
        for (int i = 0; i < asDouble.length; i++) {
            if (asDouble[i] > treshold) {
                result.add(labels.get(i));
            }
        }
        return result;
    }

    private INDArray generateINDArray(File image) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray imageVector = loader.asMatrix(image);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageVector);
        return imageVector;
    }

}
