package cz.muni.fi.image.net.core.dataset.evaluation;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataSetEvaluationCalculator implements ScoreCalculator<ComputationGraph>{
    private static Logger logger = LoggerFactory.getLogger(DataSetEvaluationCalculator.class);

    final DataSetIterator iterator;

    public DataSetEvaluationCalculator(
            DataSetIterator iterator
    ){
          this.iterator = iterator;
    }

    @Override
    public double calculateScore(ComputationGraph network) {
                iterator.reset();
        EvaluationBinary eval = new EvaluationBinary(5,null);
                network.doEvaluation(iterator,eval);
                logger.info("\n" + eval.stats() + "\n" +
                        "Actual average F1: " + eval.averageF1() + "\n" +
                        "Actual average precision: " + eval.averagePrecision() +"\n" +
                        "Actual average recall: " + eval.averageF1());
        return 1-eval.averageF1();
    }
}
