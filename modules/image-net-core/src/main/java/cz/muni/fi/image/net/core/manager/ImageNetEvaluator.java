package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.dataset.processor.DataSetTransform;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * ImageNetEvaluator serves as evaluator of trained models on custom data.
 *
 * @author Jakub Peschel
 */
public class ImageNetEvaluator {

    protected final Configuration conf;
    protected final NeuralNetModelWrapper modelWrapper;

    /**
     * Constructor for {@link ImageNetEvaluator}
     *
     * @param conf         Global {@link Configuration}
     * @param modelWrapper {@link NeuralNetModelWrapper} for tested modelWrapper
     */
    public ImageNetEvaluator(
            final Configuration conf,
            final NeuralNetModelWrapper modelWrapper
    ) {
        this.conf = conf;
        this.modelWrapper = modelWrapper;
    }


    /**
     * Test of the modelWrapper on selected dataSet
     *
     * @param dataSet {@link DataSet} containing data for testing
     * @return {@link String} containing information about performance and confusion matrix
     */
    public String evaluateModel(
            final DataSet dataSet
    ) {
        final DataSetTransform processor = new DataSetTransform(this.conf, modelWrapper.getType());
        final DataSetIterator iter = processor.presaveDataSetIterator(
                processor.prepareDataSetIterator(dataSet),
                modelWrapper.getType(),
                "evaluation"
        );

        final EvaluationBinary evaluationBinary = new EvaluationBinary(
                dataSet.getLabels().size(),
                null
        );

        evaluationBinary.setLabelNames(iter.getLabels());

        final Model model = modelWrapper.getModel();

        if (model instanceof MultiLayerNetwork) {
            ((MultiLayerNetwork) model).doEvaluation(iter, evaluationBinary);
        } else if (model instanceof ComputationGraph) {
            ((ComputationGraph) model).doEvaluation(iter, evaluationBinary);
        }
        return evaluationBinary.stats();
    }
}
