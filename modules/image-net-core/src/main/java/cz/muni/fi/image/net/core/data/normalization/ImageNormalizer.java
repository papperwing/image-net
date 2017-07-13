package cz.muni.fi.image.net.core.data.normalization;

import cz.muni.fi.image.net.core.enums.ModelType;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNormalizer {

    ModelType modelType;

    public ImageNormalizer() {
    }

    public ImageNormalizer(ModelType modelType) {
        this.modelType = modelType;
    }

    public DataNormalization getDataNormalization() {
        if (modelType == null) {
            throw new IllegalStateException("Default modelType was not set. " +
                    "Please use getDataNormalization(ModelType modelType) method");
        }
        return getDataNormalization(modelType);
    }

    public DataNormalization getDataNormalization(ModelType modelType) {
        switch (modelType) {
            case RESNET50:
                return new VGG16ImagePreProcessor();
            case VGG16:
                return new VGG16ImagePreProcessor();
            case LENET:
                return new VGG16ImagePreProcessor();
            default:
                return new VGG16ImagePreProcessor();
        }
    }
}
