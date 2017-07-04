package cz.muni.fi.image.net.model.creator;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;
import java.io.IOException;
import java.util.Random;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelBuilderImpl implements ModelBuilder {
    
    Logger logger = LoggerFactory.getLogger(ModelBuilderImpl.class);
    Configuration config;
    private static final String featureExtractionLayer = "fc1";
    
    public ModelBuilderImpl(Configuration config) {
        this.config = config;
    }

    /**
     * Method takes parameters and create {@link NeuralNetModel}.
     *
     * @return
     */
    public NeuralNetModel createModel(ModelType modelType, DataSet dataSet) {
        switch (modelType) {
            case VGG16:
                return createVggModel(dataSet);
            case LENET:
                return createLeNetModel(dataSet);
            case RESNET50:
                return createResnet50(dataSet);
        }
        throw new IllegalArgumentException("Unsuported model type selected.");
    }
    
    private NeuralNetModel createVggModel(DataSet dataSet) {
        int numClasses = dataSet.getLabels().size();
        try {
            TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
            logger.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
            ComputationGraph vgg16 = modelImportHelper.loadModel();
            logger.info(vgg16.summary());
            logger.debug("Number of elements: " + vgg16.params().lengthLong());

            //Decide on a fine tune configuration to use.
            //In cases where there already exists a setting the fine tune setting will
            //  override the setting for all layers that are not "frozen".
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .seed(this.config.getSeed())
                    .build();

            //Construct a new model with the intended architecture and print summary
            ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                    .fineTuneConfiguration(fineTuneConf)
                    .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                    .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                    .addLayer("predictions",
                            new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .nIn(4096).nOut(numClasses)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numClasses)))) //This weight init dist gave better results than Xavier
                            .activation(Activation.SIGMOID).build(),
                            "fc2")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();
            logger.info(vgg16Transfer.summary());
            logger.debug("Number of elements: " + vgg16Transfer.params().lengthLong());
            
            return new NeuralNetModel(
                    vgg16Transfer,
                    dataSet.getLabels(),
                    ModelType.VGG16
            );
        } catch (InvalidKerasConfigurationException ex) {
            logger.error("Invalid network configuration.", ex);
        } catch (UnsupportedKerasConfigurationException ex) {
            logger.error("Unsuported network configuration.", ex);
        } catch (IOException ex) {
            logger.error("Invalid rights.", ex);
        }
        return null;//replace with custom exception
    }
    
    public NeuralNetModel createLeNetModel(DataSet dataSet) {
        try {
            int numClasses = dataSet.getLabels().size();
            ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/home/jpeschel/images/googlenet/googlenet_architecture.json", "/home/jpeschel/images/googlenet/googlenet_weights.h5");
            
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .seed(this.config.getSeed())
                    .build();
            
            ComputationGraph transferedModel = new TransferLearning.GraphBuilder(model)
                    .fineTuneConfiguration(fineTuneConf)
                    .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                    .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                    .addLayer("predictions",
                            new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .nIn(4096).nOut(numClasses)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numClasses)))) //This weight init dist gave better results than Xavier
                            .activation(Activation.SIGMOID).build(),
                            "fc2")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();
            logger.info(transferedModel.summary());
            
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.LENET);
        } catch (IOException | UnsupportedKerasConfigurationException | InvalidKerasConfigurationException ex) {
            logger.error("Unable to load model", ex);
        }
        return null;
    }
    
    public NeuralNetModel createResnet50(DataSet dataSet) {
        ZooModel zooModel = new ResNet50(dataSet.getLabels().size(), new Random().nextInt(), 1, WorkspaceMode.SEPARATE);
        try {
            ComputationGraph zooModelOriginal = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .seed(this.config.getSeed())
                    .build();
            
            logger.info("Original model:\n" + zooModelOriginal.summary());
            String[] StringTypeArray = new String[0];
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
            
            logger.info("["+model.getNumInputArrays()+":"+ model.getNumOutputArrays()+"]");
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }
    
}
