package cz.muni.fi.image.net.core.dataset.Evaluation;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
                logger.info(eval.stats());
        return 1-eval.averagePrecision();
    }
}
