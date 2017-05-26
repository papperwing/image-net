package cz.muni.fi.imageNet.api;

import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.api.dto.DataSampleDTO;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.Label;
import cz.muni.fi.imageNet.core.objects.ModelType;
import cz.muni.fi.imageNet.core.objects.NeuralNetModel;
import cz.muni.fi.imageNet.dataset.creator.DataSetBuilder;
import cz.muni.fi.imageNet.dataset.creator.DataSetBuilderImpl;
import cz.muni.fi.imageNet.core.manager.ImageNetRunner;
import cz.muni.fi.imageNet.model.creator.ModelBuilder;
import cz.muni.fi.imageNet.model.creator.ModelBuilderImpl;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple API for generating and computing on Image Neural Network.
 *
 * @author Jakub Peschel
 */
public class ImageNetAPI {

    private final Logger logger = LoggerFactory.getLogger(getClass());
    private final Configuration config;

    public ImageNetAPI() {
        logger.info("Created API for ImageNet with default Configuration");
        this.config = new Configuration();
    }

    public ImageNetAPI(Configuration config) {
        logger.info("Created API for ImageNet with custom Configuration");
        this.config = config;
    }

    /**
     *
     * @param modelName
     * @param dataSamples
     * @param outputSize
     * @param modelType
     * @return
     * @throws IOException
     */
    public File getModel(String modelName, DataSampleDTO[] dataSamples, int outputSize, ModelType modelType) throws IOException {

        final long startTime = System.currentTimeMillis();

        logger.info("Starting to build model: " + modelName);

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        logger.info("Process initialized.");

        final DataSampleProcessor processor = new DataSampleProcessor(config);

        final DataSet dataSet = datasetBuilder.buildDataSet(
                processor.getDataSampleCollection(dataSamples),
                processor.getDataSampleLabels(dataSamples)
        );
        logger.info("Prepared dataset.");
        logger.debug(dataSet.getLabels().toString());

        final NeuralNetModel model = modelBuilder.createModel(
                modelType,
                dataSet
        );
        logger.info("Created model.");

        runner.trainModel(
                model,
                dataSet,
                startTime
        );
        logger.info("Trained model.");

        return model.toFile(modelName);
    }

    public File continueTraining(File modelFile, DataSampleDTO[] dataSamples, int outputSize, ModelType modelType) throws IOException {

        logger.info("Initialization of managers.");
        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);
        final ImageNetRunner runner = new ImageNetRunner(config);
        final DataSampleProcessor processor = new DataSampleProcessor(config);
        
        logger.info("Preparation of dataset.");
        final DataSet dataSet = datasetBuilder.buildDataSet(
                processor.getDataSampleCollection(dataSamples),
                processor.getDataSampleLabels(dataSamples)
        );

        logger.info("Loading of model.");
        final NeuralNetModel model = new NeuralNetModel(modelFile, dataSet.getLabels(), modelType);
        
        logger.info("Training of model.");
        runner.trainModel(model, dataSet, outputSize);

        logger.info("Saving of model.");
        return model.toFile(modelFile.getAbsolutePath());
    }

    /**
     *
     * @param modelLoc
     * @param dataSamples
     * @param classNames
     * @param modelType
     * @return
     * @throws IOException
     */
    public String evaluateModel(File modelLoc, DataSampleDTO[] dataSamples, List<String> classNames, ModelType modelType) throws IOException {

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        List<Label> labels = new ArrayList();
        for (String name : classNames) {
            labels.add(new Label(name));
        }

        NeuralNetModel model = new NeuralNetModel(
                modelLoc,
                labels,
                modelType
        );

        DataSampleProcessor processor = new DataSampleProcessor(config);
        final Collection<DataSample> dataSampleCollection = processor.getDataSampleCollection(dataSamples);

        DataSet dataset = datasetBuilder.buildDataSet(dataSampleCollection, labels);
        return runner.evaluateModel(model, dataset);
    }

    /**
     *
     *
     * @param modelLoc
     * @param labelNameList
     * @param imageURI
     * @return
     * @throws IOException
     */
    public List<List<String>> classify(String modelLoc, List<String> labelNameList, String... imageURI) throws IOException {
        final ImageNetRunner runner = new ImageNetRunner(config);
        List<Label> labelList = new ArrayList<Label>();
        for (String labelName : labelNameList) {
            labelList.add(new Label(labelName));
        }
        NeuralNetModel model = new NeuralNetModel(new File(modelLoc), labelList, ModelType.VGG16);

        return getLabelNames(runner.classify(model, imageURI));
    }

    private List<List<String>> getLabelNames(List<List<Label>> labelLists) {
        List<List<String>> results = new ArrayList();
        for (List<Label> labels : labelLists) {
            List<String> result = new ArrayList();

            for (Label label : labels) {
                result.add(label.getLabelName());
            }
            results.add(result);
        }
        return results;
    }

}
