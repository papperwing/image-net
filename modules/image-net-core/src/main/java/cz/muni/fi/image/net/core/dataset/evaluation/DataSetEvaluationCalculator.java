package cz.muni.fi.image.net.core.dataset.evaluation;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataSetEvaluationCalculator implements ScoreCalculator<ComputationGraph> {
    private static Logger logger = LoggerFactory.getLogger(DataSetEvaluationCalculator.class);

    final DataSetIterator testIterator;
    final DataSetIterator trainIterator;

    public DataSetEvaluationCalculator(
            DataSetIterator testIterator,
            DataSetIterator trainIterator
    ) {
        this.testIterator = testIterator;
        this.trainIterator = trainIterator;
    }

    @Override
    public double calculateScore(ComputationGraph network) {
        testIterator.reset();
        trainIterator.reset();

        EvaluationBinary testEval = new EvaluationBinary(5, null);
        network.doEvaluation(testIterator, testEval);
        logger.info("\n" + "Test Stats:\n" +
                testEval.stats() + "\n" +
                "Actual average F1: " + testEval.averageF1() + "\n" +
                "Actual average precision: " + testEval.averagePrecision() + "\n" +
                "Actual average recall: " + testEval.averageRecall());

        EvaluationBinary trainEval = new EvaluationBinary(5, null);
        network.doEvaluation(trainIterator, trainEval);
        trainIterator.reset();
        logger.info("\n" + "Train Stats:\n" +
                trainEval.stats() + "\n" +
                "Actual average F1: " + trainEval.averageF1() + "\n" +
                "Actual average precision: " + trainEval.averagePrecision() + "\n" +
                "Actual average recall: " + trainEval.averageRecall());

        return 1 - testEval.averageF1();
    }
}
