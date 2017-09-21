
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class App {

    public static Logger logger = LoggerFactory.getLogger(App.class);

    static String PARENT_DIR_PATH = "images";
    static int HEIGHT = 224;
    static int WIDTH = 224;
    static int BATCH_SIZE = 32;

    public static void main(String[] args) throws Exception {
        logger.info("Starting prototype of binary classification network");

        switch(args.length){
            case 1:
                switch (args[0]){
                    case "train":
                        logger.info("Starting training model");
                        App.train();
                        break;
                    case "eval":
                        logger.info("Starting evaluation of model");
                        App.eval();
                        break;
                    default:
                        logger.error("Unknown option was used.");
                        break;
                }
                break;
            default:
                logger.error("Wrong number of attributes was used.");
                break;
        }

    }
    public static void eval() throws Exception{
        ComputationGraph model = ModelSerializer.restoreComputationGraph("bestGraph.bin");
        logger.debug("Loaded model.");

        Random rng = new Random();

        File parentDir = new File(PARENT_DIR_PATH);

        FileSplit fileSplit = new FileSplit(
                parentDir,
                NativeImageLoader.ALLOWED_FORMATS,
                rng
        );

        logger.debug("Created split.");

        BalancedPathFilter pathFilter = new BalancedPathFilter(
                rng,
                NativeImageLoader.ALLOWED_FORMATS,
                new MyPathGenerator()
        );

        logger.debug("Created balancer.");

        final List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(
                new Pair(
                        new ResizeImageTransform(
                                300,
                                300
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new RandomCropTransform(
                                HEIGHT,
                                WIDTH
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new FlipImageTransform(
                                1
                        ),
                        0.5
                )
        );

        final ImageTransform combinedTransform = new MultiImageTransform(
                new ImageTransform[]{
                        new PipelineImageTransform(
                                pipeline,
                                false
                        )
                }
        );

        logger.debug("Created combined transformation.");

        RecordReader evalReader = new ImageRecordReader(
                HEIGHT,
                WIDTH,
                3,
                new MyPathGenerator(),
                combinedTransform
        );

        evalReader.initialize(fileSplit);

        logger.debug("Created reader.");

        DataSetIterator evalIterator = new RecordReaderDataSetIterator(
                evalReader,
                BATCH_SIZE,
                1,
                1,
                true
        );

        logger.debug("Created iterator.");

        Evaluation evaluation = model.evaluate(evalIterator);

        logger.info(evaluation.stats());

    }

    public static void train()throws Exception{
        ZooModel zooModel = new ResNet50(
                1,
                new Random().nextInt(),
                1,
                WorkspaceMode.SEPARATE
        );

        ComputationGraph model = (ComputationGraph) (zooModel.initPretrained());

        FineTuneConfiguration tuneConfiguration = new FineTuneConfiguration.Builder()
                .updater(Updater.ADAM)
                .learningRate(0.001)
                .build();

        ComputationGraph newModel = new TransferLearning.GraphBuilder(model)
                .removeVertexKeepConnections("fc1000")
                .fineTuneConfiguration(tuneConfiguration)
                .addLayer("fc1000",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2048)
                                .nOut(1)
                                .weightInit(WeightInit.XAVIER)
                                .biasInit(0.1)
                                .dropOut(0)
                                .activation(Activation.SIGMOID)
                                .build(),
                        "flatten_3")
                .setOutputs("fc1000")
                .build();

        logger.debug(newModel.summary());

        Random rng = new Random();

        File parentDir = new File(PARENT_DIR_PATH);

        FileSplit fileSplit = new FileSplit(
                parentDir,
                NativeImageLoader.ALLOWED_FORMATS,
                rng
        );

        BalancedPathFilter pathFilter = new BalancedPathFilter(
                rng,
                NativeImageLoader.ALLOWED_FORMATS,
                new MyPathGenerator()
        );

        InputSplit[] splits = fileSplit.sample(pathFilter, 0.8, 0.2);
        InputSplit trainSplit = splits[0];
        InputSplit testSplit = splits[1];

        final List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(
                new Pair(
                        new ResizeImageTransform(
                                300,
                                300
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new RandomCropTransform(
                                HEIGHT,
                                WIDTH
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new FlipImageTransform(
                                1
                        ),
                        0.5
                )
        );

        final ImageTransform combinedTransform = new MultiImageTransform(
                new ImageTransform[]{
                        new PipelineImageTransform(
                                pipeline,
                                false
                        )
                }
        );
        RecordReader trainReader = new ImageRecordReader(
                HEIGHT,
                WIDTH,
                3,
                new MyPathGenerator(),
                combinedTransform
        );

        trainReader.initialize(trainSplit);

        RecordReader testReader = new ImageRecordReader(
                HEIGHT,
                WIDTH,
                3,
                new MyPathGenerator(),
                combinedTransform
        );

        testReader.initialize(testSplit);

        DataSetIterator trainIterator = new RecordReaderDataSetIterator(
                trainReader,
                BATCH_SIZE,
                1,
                1,
                true
        );

        DataSetIterator testIterator = new RecordReaderDataSetIterator(
                testReader,
                BATCH_SIZE,
                1,
                1,
                true
        );

        trainIterator.setPreProcessor(new ImagePreProcessingScaler(-1, 1));
        testIterator.setPreProcessor(new ImagePreProcessingScaler(-1, 1));

        EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(300))
                .scoreCalculator(new DataSetLossCalculatorCG(testIterator, true))
                .modelSaver(new LocalFileGraphSaver("."))
                .build();

        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(
                esConfig,
                newModel,
                trainIterator
        );

        StatsStorage statsStorage = new FileStatsStorage(new File("stat_storage"));
        newModel.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
    }
}

class MyPathGenerator implements PathLabelGenerator {

    @Override
    public Writable getLabelForPath(String path) {
        return path.contains("outside") ? new IntWritable(1) : new IntWritable(0);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).toString());
    }
}