package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.objects.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;

/**
 * Class is used for running neural network base action:
 * <ul>
 *     <li>training model</li>
 *     <li>clasification of image</li>
 *     <li>evaluating model</li>
 * </ul>
 *
 * @author Jakub Peschel
 */
public class ImageNetRunner {

    protected final Configuration conf;

    /**
     * Constructor of {@link ImageNetRunner}
     * @param conf global {@link Configuration}
     */
    public ImageNetRunner(Configuration conf) {
        this.conf = conf;
    }

    /**
     *
     * @param model
     * @param dataset
     * @return
     */
    public NeuralNetModelWrapper trainModel(
            final NeuralNetModelWrapper model,
            final DataSet dataset
    ) {
        final ImageNetTrainer trainer = new ImageNetTrainer(this.conf, model);
        return trainer.trainModel(dataset);
    }

    /**
     *
     * @param modelWrapper
     * @param imageLocations
     * @return
     */
    public List<List<Label>> classify(
            final NeuralNetModelWrapper modelWrapper,
            final String[] imageLocations
    ) {

        final ImageNetClassifier classifier = new ImageNetClassifier(modelWrapper);
        return classifier.classify(imageLocations);
    }

    /**
     *
     *
     * @param modelWrapper
     * @param dataSet
     * @return
     */
    public String evaluateModel(
            final NeuralNetModelWrapper modelWrapper,
            final DataSet dataSet
    ) {
        final ImageNetEvaluator evaluator = new ImageNetEvaluator(this.conf, modelWrapper);
        return evaluator.evaluateModel(dataSet);
    }

}
