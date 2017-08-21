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
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelBuilderImpl implements ModelBuilder {

    Logger logger = LoggerFactory.getLogger(ModelBuilderImpl.class);
    private final Configuration config;

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
            /*case VGG16:
                return createVggModel(dataSet);*/
            case LENET:
                return createLeNetModel(dataSet);
            case RESNET50:
                return createResnet50(dataSet);
            case ALEXNET:
                return createAlexNet(dataSet);
        }
        throw new IllegalArgumentException("Unsuported model type selected.");
    }

    private NeuralNetModel createAlexNet(DataSet dataSet) {
        ZooModel zooModel = new AlexNet(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                1,
                WorkspaceMode.SEPARATE
        );

        MultiLayerNetwork zooModelOriginal = (MultiLayerNetwork) zooModel.init();
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(this.config.getLearningRate())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(this.config.getSeed())
                .dropOut(this.config.getDropout())
                .l1(this.config.getL1())
                .l2(this.config.getL2())
                .build();

        MultiLayerNetwork model = new TransferLearning.Builder(zooModelOriginal)
                .fineTuneConfiguration(fineTuneConf)
                .removeLayersFromOutput(2)
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(4096)
                        .nOut(dataSet.getLabels().size())
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(
                                new NormalDistribution(
                                        0,
                                        0.2*(2.0/(4096+dataSet.getLabels().size()))
                                )
                        )
                        .build())//nonlinearity layer
                .build();


        logger.info("New model:\n" + model.summary());
        return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
    }

    public NeuralNetModel createLeNetModel(DataSet dataSet) {
        ZooModel zooModel = new LeNet(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                1,
                WorkspaceMode.SEPARATE
        );
        try {
            MultiLayerNetwork zooModelOriginal = (MultiLayerNetwork) zooModel.initPretrained(PretrainedType.MNIST);
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .learningRate(this.config.getLearningRate())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.SGD)
                    .seed(this.config.getSeed())
                    .build();

            MultiLayerNetwork model = new TransferLearning.Builder(zooModelOriginal)
                    .setFeatureExtractor(8)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeOutputLayer()
                    .addLayer(new DenseLayer.Builder()
                            .nIn(500)
                            .nOut(250)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .dist(
                                    new NormalDistribution(
                                            0,
                                            0.2*(2.0/(500+dataSet.getLabels().size()))
                                    )
                            )
                            .build())//nonlinearity layer
                    .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .nIn(250)
                            .nOut(dataSet.getLabels().size())
                            .activation(Activation.SIGMOID)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .dist(
                                    new NormalDistribution(
                                            0,
                                            0.2*(2.0/(250+dataSet.getLabels().size()))
                                    )
                            )
                            .build())
                    .build();


            logger.info("New model:\n" + model.summary());
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }

    public NeuralNetModel createResnet50(DataSet dataSet) {
        ZooModel zooModel = new ResNet50(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                1,
                WorkspaceMode.SEPARATE
        );
        try {
            ComputationGraph zooModelOriginal = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
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
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(
                                            new NormalDistribution(
                                                    0,
                                                    0.2*(2.0/(2048+dataSet.getLabels().size()))
                                            )
                                    )
                                    .build(),
                            "flatten_3"
                    )
                    .addLayer("fc1000",
                            new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                    .nIn(1024)
                                    .nOut(dataSet.getLabels().size())
                                    .activation(Activation.SIGMOID)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(
                                            new NormalDistribution(
                                                    0,
                                                    0.2*(2.0/(1024+dataSet.getLabels().size()))
                                            )
                                    )
                                    .build(),
                            "nonlinearity"
                    )
                    .setOutputs("fc1000")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            logger.info("New model:\n" + model.summary());
            return new NeuralNetModel(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }

}
