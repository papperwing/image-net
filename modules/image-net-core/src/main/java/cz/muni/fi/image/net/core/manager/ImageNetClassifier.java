package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * ImageNetClassifier serves as categorization tool based on trained modelWrapper passed to it.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNetClassifier {


    private final Logger logger = LoggerFactory.getLogger(ImageNetRunner.class);


    private final double treshold = 0.5;

    protected final NeuralNetModelWrapper modelWrapper;

    /**
     * Constructor of {@link ImageNetClassifier}
     *
     * @param modelWrapper
     */
    public ImageNetClassifier(
            final NeuralNetModelWrapper modelWrapper
    ) {
        this.modelWrapper = modelWrapper;
    }

    /**
     * @param imageLocations
     * @return
     */
    public List<List<Label>> classify(
            final String[] imageLocations
    ) {

        try {
            final List<INDArray> images = new ArrayList<>();
            for (final String imageLocation : imageLocations) {
                images.add(generateINDArray(new File(imageLocation)));
            }

            final Model model = modelWrapper.getModel();
            final List<INDArray> outputArray = new ArrayList<>();
            for (final INDArray input : images) {
                if (model instanceof MultiLayerNetwork) {
                    outputArray.add(((MultiLayerNetwork) model).output(input));
                } else {
                    outputArray.add(((ComputationGraph) model).outputSingle(input));

                }
            }
            INDArray[] indArrays = new INDArray[outputArray.size()];
            indArrays = outputArray.toArray(indArrays);
            return getLabel(
                    new ArrayList(modelWrapper.getLabels()),
                    indArrays
            );
        } catch (IOException ex) {
            logger.error("Loading of image was not sucessfull.", ex);
        }
        return null;
    }


    private List<List<Label>> getLabel(

            final List<Label> labels,
            final INDArray... outputs
    ) {
        //TODO: Improve to iterate over INDArray instead of double field
        final List<List<Label>> results = new ArrayList();
        for (INDArray output : outputs) {
            final List<Label> result = new ArrayList();
            final double[] asDouble = output.dup().data().asDouble();
            for (int i = 0; i < asDouble.length; i++) {
                if (asDouble[i] > treshold) {
                    result.add(labels.get(i));
                }
            }
            results.add(result);
        }
        return results;
    }

    //TODO: finish refactor
    private INDArray generateINDArray(
            final File image
    ) throws IOException {
        final NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        final INDArray imageVector = loader.asMatrix(image);
        final DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(imageVector);
        return imageVector;
    }
}
