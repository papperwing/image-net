package cz.muni.fi.image.net.model.creator;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelBuilderImpl implements ModelBuilder {

    Map<String, String> extractionNames;

    Logger logger = LoggerFactory.getLogger(ModelBuilderImpl.class);
    Configuration config;

    public ModelBuilderImpl(Configuration config) {
        this.config = config;
        extractionNames = new HashMap<>();
        extractionNames.put(ModelType.RESNET50.toString() + PretrainedType.IMAGENET.toString(), "flatten_3");
    }

    /**
     * Method takes parameters and create {@link NeuralNetModel}.
     *
     * @return
     */
    public NeuralNetModel createModel(ModelType modelType, DataSet dataSet) {
        switch (modelType) {
            /*case VGG16:
                return createVggModel(dataSet);*/
            case LENET:
                return createLeNetModel(dataSet);
            case RESNET50:
                return createResnet50(dataSet);
        }
        throw new IllegalArgumentException("Unsuported model type selected.");
    }

    public NeuralNetModel createLeNetModel(DataSet dataSet) {
        ZooModel zooModel = new LeNet(
                dataSet.getLabels().size(),
                new Random().nextLong(),
                1,
                WorkspaceMode.SEPARATE
        );
        try {
            ComputationGraph zooModelOriginal = (ComputationGraph) zooModel.initPretrained(PretrainedType.MNIST);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .seed(this.config.getSeed())
                    .build();

            ComputationGraph model = new TransferLearning.GraphBuilder(zooModelOriginal)
                    .setFeatureExtractor("flatten_3")
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("fc1000")
                    .addLayer("nonlinearity",
                            new DenseLayer.Builder()
                                    .nIn(2048)
                                    .nOut(1024)
                                    .activation(Activation.RELU)
                                    .weightInit(WeightInit.DISTRIBUTION).build(),
                            "flatten_3"
                    )
                    .addLayer("fc1000",
                            new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                    .nIn(1024)
                                    .nOut(dataSet.getLabels().size())
                                    .activation(Activation.SIGMOID)
                                    .weightInit(WeightInit.DISTRIBUTION).build(),
                            "nonlinearity"
                    )
                    .setOutputs("fc1000")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();


            logger.info("New model:\n" + zooModelOriginal.summary());
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }

    public NeuralNetModel createResnet50(DataSet dataSet) {
        ZooModel zooModel = getZooModel(ModelType.RESNET50, dataSet.getLabels().size());
        try {
            ComputationGraph zooModelOriginal =
                    (ComputationGraph) zooModel.initPretrained(
                            getAvailablePretrainedType(modelType)
                    );
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .seed(this.config.getSeed())
                    .build();

            ComputationGraph model = new TransferLearning.GraphBuilder(zooModelOriginal)
                    .setFeatureExtractor("flatten_3")
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("fc1000")
                    .addLayer("nonlinearity",
                            new DenseLayer.Builder()
                                    .nIn(2048)
                                    .nOut(1024)
                                    .activation(Activation.RELU)
                                    .weightInit(WeightInit.DISTRIBUTION).build(),
                            "flatten_3"
                    )
                    .addLayer("fc1000",
                            new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                    .nIn(1024)
                                    .nOut(dataSet.getLabels().size())
                                    .activation(Activation.SIGMOID)
                                    .weightInit(WeightInit.DISTRIBUTION).build(),
                            "nonlinearity"
                    )
                    .setOutputs("fc1000")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            logger.info("New model:\n" + zooModelOriginal.summary());
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }

    private Model getFineTunedModel(
            ZooModel zooModel,
            ModelType modelType,
            int outputSize
    ) throws IOException {
        ComputationGraph originalModel =
                (ComputationGraph) zooModel.initPretrained(
                        getAvailablePretrainedType(modelType)
                );
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(this.config.getLearningRate())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(this.config.getSeed())
                .build();

        ComputationGraph model = new TransferLearning.GraphBuilder(originalModel)
                .setFeatureExtractor(
                        getFeatureExtractionLayer(
                                modelType,
                                getAvailablePretrainedType(modelType)
                        )
                )
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("fc1000")
                .addLayer("nonlinearity",
                        new DenseLayer.Builder()
                                .nIn(2048)
                                .nOut(1024)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.DISTRIBUTION).build(),
                        "flatten_3"
                )
                .addLayer("fc1000",
                        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                .nIn(1024)
                                .nOut(outputSize)
                                .activation(Activation.SIGMOID)
                                .weightInit(WeightInit.DISTRIBUTION).build(),
                        "nonlinearity"
                )
                .setOutputs("fc1000")
                .setWorkspaceMode(WorkspaceMode.SEPARATE)
                .build();

        logger.info("New model:\n" + originalModel.summary());
        return model;
    }

    private String getFeatureExtractionLayer(ModelType modelType, PretrainedType pretrainedType) {
        return extractionNames.get(modelType.toString() + pretrainedType.toString());
    }

    private ZooModel getZooModel(ModelType modelType, int outputSize) {
        switch (modelType) {
            case RESNET50:
                return new ResNet50(
                        outputSize,
                        new Random().nextLong(),
                        1,
                        WorkspaceMode.SEPARATE
                );
            case LENET:
                return new LeNet(
                        outputSize,
                        new Random().nextLong(),
                        1,
                        WorkspaceMode.SEPARATE
                );
            case VGG16:
                return new VGG16(
                        outputSize,
                        new Random().nextLong(),
                        1,
                        WorkspaceMode.SEPARATE
                );
            case INCEPTIONV1:
                return new InceptionResNetV1(
                        outputSize,
                        new Random().nextLong(),
                        1,
                        WorkspaceMode.SEPARATE
                );
        }
        throw new IllegalArgumentException("Selected model type isn't available");
    }


    private PretrainedType getAvailablePretrainedType(ModelType type) {
        switch (type) {
            case RESNET50:
                return PretrainedType.MNIST;
            case LENET:
                return PretrainedType.MNIST;
        }
        throw new IllegalArgumentException("Selected model type isn't available");

    }

}
