import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class App {

    public static Logger logger = LoggerFactory.getLogger(App.class);

    static String PARENT_DIR_PATH = "images";
    static int HEIGHT = 224;
    static int WIDTH = 224;
    static int BATCH_SIZE = 16;

    public static void main(String[] args) throws Exception {
        try {
            logger.info("Starting prototype of binary classification network");

            switch (args.length) {
                case 1:
                    switch (args[0]) {
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
        } catch (Exception ex) {
            logger.error("Error occured.", ex);
        }
    }

    public static void eval() throws Exception {
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

        RecordReader evalReader = new CustomImageRecordReader(
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

        evalIterator.setPreProcessor(new ImagePreProcessingScaler(-1, 1));

        logger.debug("Created iterator.");

        Evaluation evaluation = new Evaluation(1) {
            @Override
            public String stats() {
                StringBuilder builder = new StringBuilder();
                builder.append("TP|TN|FP|FN\n");
                builder.append(
                        truePositives.getCount(0) + " | " +
                                trueNegatives.getCount(0) + " | " +
                                falsePositives.getCount(0) + " | " +
                                falseNegatives.getCount(0) + "\n"
                );
                builder.append(super.stats());
                return builder.toString();
            }
        };
        model.doEvaluation(evalIterator, evaluation);

        logger.info(evaluation.stats());

    }

    public static void train() throws Exception {
        ZooModel zooModel = new ResNet50(
                1,
                new Random().nextInt(),
                1,
                WorkspaceMode.SEPARATE
        );

        ComputationGraph model = (ComputationGraph) (zooModel.initPretrained());

        FineTuneConfiguration tuneConfiguration = new FineTuneConfiguration.Builder()
                .updater(Updater.NESTEROVS)
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

        logger.debug("Created splits");
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

        logger.debug("Created transformation");
        RecordReader trainReader = new CustomImageRecordReader(
                HEIGHT,
                WIDTH,
                3,
                new MyPathGenerator(),
                combinedTransform
        );

        trainReader.initialize(trainSplit);

        RecordReader testReader = new CustomImageRecordReader(
                HEIGHT,
                WIDTH,
                3,
                new MyPathGenerator(),
                combinedTransform
        );

        testReader.initialize(testSplit);

        logger.debug("Created readers");

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

        logger.debug("Prepared iterators");

        EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(300))
                .scoreCalculator(new DataSetLossCalculatorCG(testIterator, true))
                .modelSaver(new LocalFileGraphSaver("."))
                .build();

        logger.debug("Created EarlyStoppingConfig");
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(
                esConfig,
                newModel,
                trainIterator
        );

        StatsStorage statsStorage = new FileStatsStorage(new File("stat_storage"));
        newModel.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

        trainer.fit();
    }
}

class MyPathGenerator implements PathLabelGenerator {

    @Override
    public Writable getLabelForPath(String path) {
        return new File(path).getParentFile().getName().contains("outside") ? new IntWritable(1) : new IntWritable(0);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).toString());
    }
}

class CustomImageRecordReader extends ImageRecordReader {
    /**
     * Loads images with given height, width, and channels, appending labels returned by the generator.
     */
    public CustomImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator,
                                   ImageTransform imageTransform) {
        super(height, width, channels, labelGenerator, imageTransform);
    }

    @Override
    public List<Writable> next() {
        if (iter != null) {
            List<Writable> ret;
            File image = iter.next();
            currentFile = image;

            if (image.isDirectory())
                return next();
            try {
                invokeListeners(image);
                INDArray row = imageLoader.asMatrix(image);
                Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE);
                ret = RecordConverter.toRecord(row);
                if (appendLabel || writeLabel)
                    ret.add(labelGenerator.getLabelForPath(image.getPath()));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return ret;
        } else if (record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    @Override
    public List<Writable> next(int num) {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }

        List<File> currBatch = new ArrayList<>();

        int cnt = 0;

        int numCategories = (appendLabel || writeLabel) ? labels.size() : 0;
        List<Integer> currLabels = new ArrayList<>();
        while (cnt < num && iter.hasNext()) {
            File currFile = iter.next();
            currBatch.add(currFile);
            if (appendLabel || writeLabel)
                currLabels.add(labelGenerator.getLabelForPath(currFile.getPath()).toInt());
            cnt++;
        }

        INDArray features = Nd4j.createUninitialized(new int[]{cnt, channels, height, width}, 'c');
        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST);
        for (int i = 0; i < cnt; i++) {
            try {
                ((NativeImageLoader) imageLoader).asMatrixView(currBatch.get(i),
                        features.tensorAlongDimension(i, 1, 2, 3));
            } catch (Exception e) {
                System.out.println("Image file failed during load: " + currBatch.get(i).getAbsolutePath());
                throw new RuntimeException(e);
            }
        }
        Nd4j.getAffinityManager().ensureLocation(features, AffinityManager.Location.DEVICE);


        List<Writable> ret = (RecordConverter.toRecord(features));
        if (appendLabel || writeLabel) {
            INDArray labels = Nd4j.create(cnt, numCategories, 'c');
            Nd4j.getAffinityManager().tagLocation(labels, AffinityManager.Location.HOST);
            for (int i = 0; i < currLabels.size(); i++) {
                labels.putScalar(i, 0, currLabels.get(i));
            }
            ret.add(new NDArrayWritable(labels));
        }

        return ret;
    }

    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        inputSplit = split;
        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (appendLabel) {
                for (URI location : locations) {
                    File imgFile = new File(location);
                    File parentDir = imgFile.getParentFile();
                    String name = parentDir.getName();
                    if (labelGenerator != null) {
                        name = "outside";
                    }
                    if (!labels.contains(name)) {
                        labels.add(name);
                    }
                    if (pattern != null) {
                        String label = name.split(pattern)[patternPosition];
                        fileNameMap.put(imgFile.toString(), label);
                    }
                }
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
            //            iter = new FileFromPathIterator(allPaths.iterator()); //This handles randomization internally if necessary
        } else
            throw new IllegalArgumentException("No path locations found in the split.");

        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        Collections.sort(labels);
    }

}