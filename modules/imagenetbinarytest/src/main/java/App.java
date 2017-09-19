import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class App {

    public static Logger logger = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws Exception {
        logger.info("Starting prototype of binary classification network");
        String PARENT_DIR = "./images";

        ZooModel zooModel = new ResNet50(1, new Random().nextInt(),1, WorkspaceMode.SEPARATE);
        ComputationGraph model = (ComputationGraph)(zooModel.initPretrained());
        logger.debug(model.summary());

        FineTuneConfiguration tuneConfiguration = new FineTuneConfiguration.Builder()
                .updater(Updater.ADAM)
                .learningRate(0.001)
                .build();

        ComputationGraph newModel = new TransferLearning.GraphBuilder(model)
                .removeVertexKeepConnections("fc1000")
                .fineTuneConfiguration(tuneConfiguration)
                .addLayer("fc1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2048)
                        .nOut(1)
                        .weightInit(WeightInit.XAVIER)
                        .biasInit(0.1)
                        .dropOut(0)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
    }
}
