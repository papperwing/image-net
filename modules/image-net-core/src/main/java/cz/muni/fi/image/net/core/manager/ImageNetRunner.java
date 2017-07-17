package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.data.normalization.ImageNormalizer;
import cz.muni.fi.image.net.core.dataset.processor.DataSetProcessor;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.EarlyStoppingParallelTrainer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Class is used for running neural network base action:
 * <ul><li>training model</li><li>clasification</li></ul>
 *
 * @author Jakub Peschel
 */
public class ImageNetRunner {

    private final Logger logger = LoggerFactory.getLogger(ImageNetRunner.class);


    private final double treshold = 0.5;
    private final double splitPercentage = 0.8;

    protected final Configuration conf;

    /**
     * @param conf
     */
    public ImageNetRunner(Configuration conf) {
        this.conf = conf;
    }

    /**
     * @param model
     * @param dataset
     * @return
     */
    public NeuralNetModel trainModel(
            final NeuralNetModel model,
            final DataSet dataset
    ) {
        ImageNetTrainer trainer = new ImageNetTrainer(this.conf);
        return trainer.trainModel(model, dataset);
    }

    /**
     * @param modelWrapper
     * @param imageLocations
     * @return
     */
    public List<List<Label>> classify(
            final NeuralNetModel modelWrapper,
            final String[] imageLocations
    ) {

        ImageNetClassifier classifier = new ImageNetClassifier(this.conf);
        return classifier.classify(modelWrapper, imageLocations);
    }

    public String evaluateModel(
            NeuralNetModel modelWrapper,
            DataSet dataSet
    ) {
        ImageNetEvaluator evaluator = new ImageNetEvaluator(this.conf);
        return evaluator.evaluateModel(modelWrapper, dataSet);
    }

}
