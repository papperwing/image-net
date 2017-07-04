package cz.muni.fi.imageNet.api;

import cz.muni.fi.imageNet.api.dto.DataSampleDTO;
import cz.muni.fi.imageNet.core.manager.ImageNetRunner;
import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.ModelType;
import cz.muni.fi.imageNet.core.objects.NeuralNetModel;
import cz.muni.fi.imageNet.dataset.creator.DataSetBuilder;
import cz.muni.fi.imageNet.dataset.creator.DataSetBuilderImpl;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.fail;

/**
 * Example is simplified version of whole training process for debugging purposes.
 *
 * Created by jpeschel on 3.7.17.
 */
public class WorkFlowTest {

    Logger logger = LoggerFactory.getLogger(this.getClass());
    Configuration config;

    @Test
    public void workFlowExample() throws Exception {

        DataSet dataSet = setUpDataSet();

        NeuralNetModel modelWrapper = createModel(dataSet);

        ImageNetRunner runner = new ImageNetRunner(this.config);

        runner.trainModel(modelWrapper, dataSet);

    }

    private NeuralNetModel createModel(DataSet dataSet) throws Exception {
        ZooModel zooModel = new ResNet50(dataSet.getLabels().size(), new Random().nextInt(), 1, WorkspaceMode.SEPARATE);

        ComputationGraph zooModelOriginal = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(0.001)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(4)
                .build();

        logger.info("Original model:\n" + zooModelOriginal.summary());
        ComputationGraph model = new TransferLearning.GraphBuilder(zooModelOriginal)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("fc1000")
                .addLayer("fc1000",
                        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                .nIn(2048)
                                .nOut(dataSet.getLabels().size())
                                .activation(Activation.SIGMOID)
                                .weightInit(WeightInit.DISTRIBUTION).build(),
                        "flatten_3"
                )
                //.setOutputs("fc1000") uncomment after bugfix
                .setWorkspaceMode(WorkspaceMode.SEPARATE)
                .build();

        logger.info("[" + model.getNumInputArrays() + ":" + model.getNumOutputArrays() + "]");
        return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);

    }


    /**
     * Method use downloader for downloading images into local storage
     *
     * @return DataSet loaded into memory.
     * @throws Exception
     */
    private DataSet setUpDataSet() throws Exception {
        String dataSetFileName = "data.csv";
        String dataSetLocation = ClassLoader.getSystemResource(dataSetFileName).getFile();
        logger.info("Loading of resource " + dataSetFileName);
        File dataSetFile = new File(dataSetLocation);

        List<DataSampleDTO> datasetList = new ArrayList<>();

        try (final BufferedReader fileReader = new BufferedReader(new FileReader(dataSetFile))) {
            String line = fileReader.readLine();
            while (line != null) {

                DataSampleDTO sample = new DataSampleDTO(line);
                datasetList.add(sample);
                line = fileReader.readLine();
            }
        }

        DataSampleDTO[] dataSamples = new DataSampleDTO[datasetList.size()];
        dataSamples = datasetList.toArray(dataSamples);

        Configuration configuration = new Configuration();
        configuration.setTempFolder("./tmp"); //using disk storage save memory. System tmp uses mostly memory
        this.config = configuration;
        final DataSetBuilder datasetBuilder = new DataSetBuilderImpl(configuration);
        DataSampleProcessor processor = new DataSampleProcessor(configuration);

        final DataSet dataSet = datasetBuilder.buildDataSet(
                processor.getDataSampleCollection(dataSamples),
                processor.getDataSampleLabels(dataSamples)
        );

        return dataSet;
    }

}
