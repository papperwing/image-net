package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.dataset.processor.DataSetProcessor;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author Jakub Peschel
 */
public class ImageNetEvaluator {

    protected final Configuration conf;

    public ImageNetEvaluator(Configuration conf) {
        this.conf = conf;
    }

    public String evaluateModel(
            NeuralNetModel modelWrapper,
            DataSet dataSet
    ) {
        DataSetProcessor processor = new DataSetProcessor(this.conf, modelWrapper.getType());
        DataSetIterator iter = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(dataSet),
                modelWrapper.getType(),
                "evaluation"
        );

        final EvaluationBinary evaluationBinary = new EvaluationBinary(
                dataSet.getLabels().size(),
                null
        );

        evaluationBinary.setLabelNames(iter.getLabels());

        Model model = modelWrapper.getModel();

        if (model instanceof MultiLayerNetwork) {
            ((MultiLayerNetwork) model).doEvaluation(iter, evaluationBinary);
        } else if (model instanceof ComputationGraph) {
            ((ComputationGraph) model).doEvaluation(iter, evaluationBinary);
        }
        return evaluationBinary.stats();
    }
}
