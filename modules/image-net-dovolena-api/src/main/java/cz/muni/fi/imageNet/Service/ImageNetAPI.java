package cz.muni.fi.imageNet.Service;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataImage;
import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.POJO.DataSampleDTO;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.DownloadResult;
import cz.muni.fi.imageNet.Pojo.Label;
import cz.muni.fi.imageNet.Pojo.ModelType;
import cz.muni.fi.imageNet.Pojo.NeuralNetModel;
import cz.muni.fi.imageNet.Pojo.UrlImage;
import cz.muni.fi.imageNet.enums.DownloadState;
import cz.muni.fi.imageNet.manager.DataSetBuilder;
import cz.muni.fi.imageNet.manager.DataSetBuilderImpl;
import cz.muni.fi.imageNet.manager.ImageDownloadManager;
import cz.muni.fi.imageNet.manager.ImageNetRunner;
import cz.muni.fi.imageNet.manager.ModelBuilder;
import cz.muni.fi.imageNet.manager.ModelBuilderImpl;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
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

    public File getModel(String modelName, DataSampleDTO[] dataSamples, int outputSize, ModelType modelType) throws IOException {
        
        final long startTime = System.currentTimeMillis();
        
        logger.info("Starting to build model: " + modelName);

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        logger.info("Process initialized.");
        
        DataSet dataSet = datasetBuilder.buildDataSet(
                getDataSampleCollection(dataSamples),
                getDataSampleLabels(dataSamples)
        );
        logger.info("Prepared dataset.");
        logger.debug(dataSet.getLabels().toString());
        
        NeuralNetModel model = modelBuilder.createModel(
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

    private Collection<DataSample> getDataSampleCollection(DataSampleDTO[] dataSamples) {
        List<DataSample> result = new ArrayList();
        List<UrlImage> urlImages = new ArrayList();
        ImageDownloadManager downloadManager = new ImageDownloadManager(config);
        //třízení pro stahování a pro založení datasetu
        for (DataSampleDTO dataSample : dataSamples) {
            urlImages.add(transformToUrlImage(dataSample));
        }
        result.addAll(transformToDataSamples(downloadManager.processRequest(urlImages)));
        return result;
    }

    private List<Label> getDataSampleLabels(DataSampleDTO[] dataSamples) {
        Set<Label> labels = new TreeSet<Label>();
        for (DataSampleDTO dataSample : dataSamples) {
            for (String labelName : dataSample.getLabels()) {
                labels.add(new Label(labelName));
            }
        }
        return new ArrayList(labels);
    }

    private Collection<? extends DataSample> transformToDataSamples(DownloadResult processRequest) {
        Collection<DataSample> result = new ArrayList<DataSample>();
        for (DataImage dataImage : processRequest.getDataImageList()) {
            //TODO: vyřešit problémy s nestaženými
            if (dataImage.getState() == DownloadState.DOWNLOADED
                    || dataImage.getState() == DownloadState.ON_DISK) {
                result.add(
                        new DataSample(
                                dataImage.getImage().getAbsolutePath(),
                                dataImage.getUrlImage().getLabels()
                        )
                );
            }
        }
        return result;
    }

    private UrlImage transformToUrlImage(DataSampleDTO dataSample) {
        Set<Label> labels = new TreeSet();
        for (String labelName : dataSample.getLabels()) {
            labels.add(new Label(labelName));
        }
        return new UrlImage(dataSample.getUrl(), labels);
    }
    
    public File getTestModel(String modelName, DataSampleDTO[] dataSamples, int outputSize, ModelType modelType) throws IOException {
        
        final long startTime = System.currentTimeMillis();
        
        logger.info("Starting to build model: " + modelName);

        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(config);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final ImageNetRunner runner = new ImageNetRunner(config);

        logger.info("Process initialized.");
        
        DataSet dataSet = datasetBuilder.buildDataSet(
                getDataSampleCollection(dataSamples),
                getDataSampleLabels(dataSamples)
        );
        logger.info("Prepared dataset.");
        logger.debug(dataSet.getLabels().toString());
        
        NeuralNetModel model = modelBuilder.createModel(
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
        
        return null;//model.toFile(modelName);
    }
    
    public List<String> classify(String modelLoc, List<String> labelNameList, String imageURI) throws IOException{
        final ImageNetRunner runner = new ImageNetRunner(config);
        List<Label> labelList = new ArrayList<Label>();
        for (String labelName : labelNameList){
            labelList.add(new Label(labelName));
        }
        NeuralNetModel model = new NeuralNetModel(new File(modelLoc), labelList ,ModelType.VGG16);
        return getLabelNames(runner.classify(model, imageURI));
    }

    private List<String> getLabelNames(List<Label> labels) {
        List<String> result = new ArrayList();
        for (Label label : labels){
            result.add(label.getLabelName());
        }
        return result;
    }

}
