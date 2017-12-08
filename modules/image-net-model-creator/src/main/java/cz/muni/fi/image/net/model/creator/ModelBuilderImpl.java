package cz.muni.fi.image.net.model.creator;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;

import java.io.IOException;
import java.util.*;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builder for specific model based on statistic of used {@link DataSet}.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ModelBuilderImpl implements ModelBuilder {

    Logger logger = LoggerFactory.getLogger(ModelBuilderImpl.class);
    private final Configuration config;

    /**
     * Constructor of {@link ModelBuilderImpl}
     *
     * @param config global {@link Configuration}
     */
    public ModelBuilderImpl(Configuration config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NeuralNetModelWrapper createModel(
            final ModelType modelType,
            final DataSet dataSet
    ) {
        //TODO: fix and complete other methods (only one working is ALEXNET)
        switch (modelType) {
            case VGG16:
                return createVggModel(dataSet);
            //case LENET:
            //    return createLeNetModel(dataSet);
            case RESNET50:
                return createResnet50(dataSet);
            //case ALEXNET:
            //    return createAlexNet(dataSet);
        }
        throw new IllegalArgumentException("Unsuported modelWrapper type selected.");
    }

    /**
     * Create modelWrapper for other use.
     *
     * @param dataSet Data sample set containing images and labels
     * @return {@link NeuralNetModelWrapper} containing adapted configuration of AlexNet
     */
    /*private NeuralNetModelWrapper createAlexNet(final DataSet dataSet) {
        final ZooModel zooModel = new AlexNet(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                1,
                WorkspaceMode.SEPARATE
        );

        MultiLayerNetwork zooModelOriginal = (MultiLayerNetwork) zooModel.init();
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(this.config.getLearningRate())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .seed(this.config.getSeed())
                .l1(this.config.getL1())
                .l2(this.config.getL2())
                .regularization(true)
                .useRegularization(true)
                .weightInit(WeightInit.RELU)
                .biasInit(0.0)
                .dropOut(this.config.getDropout())
                .build();


        MultiLayerNetwork model = new TransferLearning.Builder(zooModelOriginal)
                .fineTuneConfiguration(fineTuneConf)
                .removeLayersFromOutput(2)
                .addLayer(new DenseLayer.Builder()
                        .nIn(4096)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .build())
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .lossFunction(new LossBinaryXENT())
                        .nIn(1024)
                        .nOut(dataSet.getLabels().size())
                        .dropOut(0)
                        .activation(Activation.SIGMOID)
                        .l2(this.config.getOutputL2())
                        .learningRate(this.config.getOLearningRate())
                        .build())
                .build();


        logger.info("New modelWrapper:\n" + model.summary());
        return new NeuralNetModelWrapper(model, dataSet.getLabels(), ModelType.ALEXNET);
    }*/

    /**
     * Create modelWrapper for other use.
     *
     * @param dataSet Data sample set containing images and labels
     * @return {@link NeuralNetModelWrapper} containing adapted configuration of LeNet
     */
    /*public NeuralNetModelWrapper createLeNetModel(final DataSet dataSet) {
        ZooModel zooModel = new LeNet(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                1,
                WorkspaceMode.SEPARATE
        );
        try {
            MultiLayerNetwork zooModelOriginal = (MultiLayerNetwork) zooModel.initPretrained(PretrainedType.IMAGENET);
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
                                            0.2 * (2.0 / (500 + dataSet.getLabels().size()))
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
                                            0.2 * (2.0 / (250 + dataSet.getLabels().size()))
                                    )
                            )
                            .build())
                    .build();


            logger.info("New modelWrapper:\n" + model.summary());
            return new NeuralNetModelWrapper(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }*/


    /**
     * Create modelWrapper for other use.
     *
     * @param dataSet Data sample set containing images and labels
     * @return {@link NeuralNetModelWrapper} containing adapted configuration of ResNet50
     */
    public NeuralNetModelWrapper createResnet50(final DataSet dataSet) {
        ZooModel zooModel = new ResNet50(
                dataSet.getLabels().size(),
                this.config.getSeed(),
                2,
                WorkspaceMode.SEPARATE
        );

        List<Integer> values = new ArrayList<>(dataSet.getLabelDistribution().values());
        logger.info("Labels: " + values.size());
        double[] dValues = new double [values.size()];
        for (int i =0 ; i < values.size(); i++){
            dValues[i] = values.get(i).doubleValue();
        }
        INDArray ones = Nd4j.ones(values.size());
        INDArray iValues = Nd4j.create(dValues);
        INDArray normValues = Transforms.unitVec(iValues);//iValues.divi(iValues.norm1Number());
        logger.info("Normalized INDA: " + normValues);
        INDArray weights = ones.sub(normValues);
        logger.info("Subbed INDA: " + weights);

        Map<Integer,Double> lrsch = new LinkedHashMap<>();
        lrsch.put(0,this.config.getLearningRate());
        lrsch.put(500, 0.0001);
        lrsch.put(1000, 0.00005);
        lrsch.put(2000, 0.00001);
        lrsch.put(4000, 0.000008);
        lrsch.put(6000, 0.000002);
        lrsch.put(8000, 0.000001);
        lrsch.put(15000, 0.0000005);

        try {
            ComputationGraph zooModelOriginal = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    //.l1(this.config.getL1())
                    //.l2(this.config.getL2())
                    .learningRate(this.config.getLearningRate())
                    .learningRatePolicy(LearningRatePolicy.Schedule)
                    .learningRateSchedule(lrsch)
                    .dropOut(this.config.getDropout())
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(0.1)
                    .updater(new Adam.Builder().build())
                    .seed(this.config.getSeed())
                    .build();

            ComputationGraph model = new TransferLearning.GraphBuilder(zooModelOriginal)
                    .setFeatureExtractor("activation_144")
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("fc1000")
                    .addLayer("fc1000",
                            new OutputLayer.Builder(new LossBinaryXENT(weights))
                                    .nIn(2048)
                                    .nOut(dataSet.getLabels().size())
                                    .activation(Activation.SIGMOID)
                                    .weightInit(WeightInit.XAVIER)
                                    .dropOut(0)
                                    .build(),
                            "flatten_3"
                    )
                    .setOutputs("fc1000")
                    .setWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            logger.info("New modelWrapper:\n" + model.summary());
            return new NeuralNetModelWrapper(model, dataSet.getLabels(), ModelType.RESNET50);
        } catch (IOException ex) {
            throw new IllegalStateException("Model weights was not loaded", ex);
        }
    }

    private NeuralNetModelWrapper createVggModel(final DataSet dataSet) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

}
