package cz.muni.fi.image.net.api;

import cz.muni.fi.image.net.api.dto.DataSampleDTO;
import cz.muni.fi.image.net.core.manager.ImageNetRunner;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.model.creator.ModelBuilderImpl;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import cz.muni.fi.image.net.dataset.creator.DataSetBuilder;
import cz.muni.fi.image.net.dataset.creator.DataSetBuilderImpl;
import cz.muni.fi.image.net.model.creator.ModelBuilder;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple API for generating and computing on Image Neural Network.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNetAPI {

    private final Logger logger = LoggerFactory.getLogger(getClass());
    private final Configuration config;

    /**
     * Constructor of {@link ImageNetAPI} with deafult {@link Configuration}
     */
    public ImageNetAPI() {
        logger.info("Created API for ImageNet with default Configuration");
        this.config = new Configuration();
    }


    /**
     * Constructor of {@link ImageNetAPI}
     *
     * @param config global {@link Configuration}
     */
    public ImageNetAPI(Configuration config) {
        logger.info("Created API for ImageNet with custom Configuration");
        this.config = config;
    }

    /**
     * Train new model.
     *
     * @param modelName   filename under which model will be saved as {@link File}
     * @param dataSamples dataset in form of array of {@link DataSampleDTO}
     * @param modelType   type of underlying pretrained networks. {@link ModelType}
     * @return {@link File} containing saved model description and model weights.
     * @throws IOException when model cannot be stored
     */
    public File getModel(
            final String modelName,
            final DataSampleDTO[] dataSamples,
            final ModelType modelType
    ) throws IOException {

        logger.info("Starting to build modelWrapper: " + modelName);

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        logger.info("Process initialized.");

        final DataSampleTranslator processor = new DataSampleTranslator(config);

        final DataSet dataSet = datasetBuilder.buildDataSet(
                processor.getDataSampleCollection(dataSamples),
                processor.getDataSampleLabels(dataSamples)
        );
        logger.info("Prepared dataset.");
        logger.debug(dataSet.getLabels().toString());

        final NeuralNetModelWrapper model = modelBuilder.createModel(
                modelType,
                dataSet
        );
        logger.info("Created modelWrapper.");

        final NeuralNetModelWrapper trainedModel = runner.trainModel(
                model,
                dataSet
        );
        logger.info("Trained modelWrapper.");

        return trainedModel.toFile(modelName);
    }

    /**
     * Continue training with existing model.
     *
     * @param modelLoc    {@link File} containing trained model
     * @param dataSamples dataset in form of array of {@link DataSampleDTO}
     * @param modelType   type of underlying pretrained networks. {@link ModelType}
     * @return {@link File} containing saved model description and model weights.
     * @throws IOException when model cannot be stored or restored from {@link File}
     */
    public File continueTraining(
            final File modelLoc,
            final DataSampleDTO[] dataSamples,
            final ModelType modelType
    ) throws IOException {

        logger.info("Initialization of managers.");
        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);
        final ImageNetRunner runner = new ImageNetRunner(config);
        final DataSampleTranslator processor = new DataSampleTranslator(config);

        logger.info("Preparation of dataset.");
        final DataSet dataSet = datasetBuilder.buildDataSet(
                processor.getDataSampleCollection(dataSamples),
                processor.getDataSampleLabels(dataSamples)
        );

        logger.info("Loading of modelWrapper.");
        final NeuralNetModelWrapper model = new NeuralNetModelWrapper(modelLoc, dataSet.getLabels(), modelType);

        logger.info("Training of modelWrapper.");
        final NeuralNetModelWrapper trainedModel = runner.trainModel(
                model,
                dataSet
        );

        logger.info("Saving of modelWrapper.");
        return trainedModel.toFile(modelLoc.getAbsolutePath());
    }

    /**
     * Get evaluation statistics about trained model based on choosed dataset
     *
     * @param modelLoc      {@link File} containing trained model
     * @param dataSamples   dataset in form of array of {@link DataSampleDTO}
     * @param labelNameList list of label names //TODO: Store together with model in .zip
     * @param modelType     type of underlying pretrained networks. {@link ModelType}
     * @return evaluation statistic containing confusion matrix together with other descriptive statistics
     * @throws IOException when model cannot be restored from {@link File}
     */
    public String evaluateModel(
            final File modelLoc,
            final DataSampleDTO[] dataSamples,
            final List<String> labelNameList,
            final ModelType modelType
    ) throws IOException {

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        final List<Label> labels = new ArrayList();
        for (String name : labelNameList) {
            labels.add(new Label(name));
        }

        final NeuralNetModelWrapper model = new NeuralNetModelWrapper(
                modelLoc,
                labels,
                modelType
        );

        final DataSampleTranslator processor = new DataSampleTranslator(config);
        final List<DataSample> dataSampleCollection = processor.getDataSampleCollection(dataSamples);

        final DataSet dataset = datasetBuilder.buildDataSet(dataSampleCollection, labels);
        return runner.evaluateModel(model, dataset);
    }

    /**
     * Sequential classification of batch of images
     *
     * @param modelLoc      {@link File} containing trained model
     * @param labelNameList list of label names //TODO: Store together with model in .zip
     * @param imageURI      array of {@link String} locations of images //TODO: refactor to URI
     * @return {@link List} of {@link List}s of classified label names //TODO: need better representation
     * @throws IOException when model cannot be restored from {@link File}
     */
    public List<List<String>> classify(
            final String modelLoc,
            List<String> labelNameList,
            String... imageURI
    ) throws IOException {
        final ImageNetRunner runner = new ImageNetRunner(config);
        final List<Label> labelList = new ArrayList<>();
        for (final String labelName : labelNameList) {
            labelList.add(new Label(labelName));
        }
        final NeuralNetModelWrapper model = new NeuralNetModelWrapper(new File(modelLoc), labelList, ModelType.VGG16);

        return getLabelNames(runner.classify(model, imageURI));
    }

    private List<List<String>> getLabelNames(
            final List<List<Label>> labelLists
    ) {
        final List<List<String>> results = new ArrayList();
        for (final List<Label> labels : labelLists) {
            final List<String> result = new ArrayList();

            for (final Label label : labels) {
                result.add(label.getLabelName());
            }
            results.add(result);
        }
        return results;
    }
}
