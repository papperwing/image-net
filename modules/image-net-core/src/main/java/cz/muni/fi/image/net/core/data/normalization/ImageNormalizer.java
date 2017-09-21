package cz.muni.fi.image.net.core.data.normalization;

import cz.muni.fi.image.net.core.enums.ModelType;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

/**
 * Selector of suitable {@link DataNormalization}
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNormalizer {

    ModelType modelType;


    /**
     * Constructor of {@link ImageNormalizer}
     *
     * @param modelType {@link ModelType} of underlying model
     */
    public ImageNormalizer(ModelType modelType) {
        this.modelType = modelType;
    }

    /**
     * Getter of suitable {@link DataNormalization}
     */
    public DataNormalization getDataNormalization() {
        switch (modelType) {
            case RESNET50:
                return new ImagePreProcessingScaler(-1, 1);
            case VGG16:
                return new VGG16ImagePreProcessor();
            case LENET:
                return new VGG16ImagePreProcessor();
            case ALEXNET:
                return new ImagePreProcessingScaler(-1, 1);
            default:
                return new ImagePreProcessingScaler(-1, 1);
        }
    }
}
