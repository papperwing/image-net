package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.objects.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

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
    public NeuralNetModelWrapper trainModel(
            final NeuralNetModelWrapper model,
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
            final NeuralNetModelWrapper modelWrapper,
            final String[] imageLocations
    ) {

        ImageNetClassifier classifier = new ImageNetClassifier(this.conf);
        return classifier.classify(modelWrapper, imageLocations);
    }

    public String evaluateModel(
            NeuralNetModelWrapper modelWrapper,
            DataSet dataSet
    ) {
        ImageNetEvaluator evaluator = new ImageNetEvaluator(this.conf);
        return evaluator.evaluateModel(modelWrapper, dataSet);
    }

}
