package cz.muni.fi.image.net.core.objects;

import java.io.File;
import java.io.IOException;
import java.util.List;

import cz.muni.fi.image.net.core.enums.ModelType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;

/**
 * Class for storing modelWrapper of neural net
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class NeuralNetModelWrapper {

    private final Model model;

    private final List<Label> labels;

    private final ModelType type;

    private String description;

    private boolean trained;


    public NeuralNetModelWrapper(
            final Model model,
            final List<Label> labelList,
            final ModelType type
    ) {
        this.trained = false;
        this.labels = labelList;
        this.model = model;
        this.type = type;
    }

    public NeuralNetModelWrapper(
            final File savedModel,
            final List<Label> labelList,
            final ModelType type
    ) throws IOException {
        this.trained = false;
        if (isModelCG(type)) {
            this.model = ModelSerializer.restoreComputationGraph(savedModel);
        } else {
            this.model = ModelSerializer.restoreMultiLayerNetwork(savedModel);
        }
        this.labels = labelList;
        this.type = type;
    }

    public Model getModel() {
        return model;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(final String description) {
        this.description = description;
    }

    public List<Label> getLabels() {
        return labels;
    }

    public File toFile(final String storeFileLocation) throws IOException {
        final String fileNameWithExtension = storeFileLocation + ".zip";
        final File saveFile = new File(fileNameWithExtension);
        ModelSerializer.writeModel(model, saveFile, trained);
        return saveFile;
    }

    public boolean isTrained() {
        return trained;
    }

    public void setTrained() {
        this.trained = true;
    }

    public ModelType getType() {
        return type;
    }

    private Boolean isModelCG(final ModelType type) {
        switch (type) {
            case RESNET50:
                return true;
            case ALEXNET:
                return false;
            case LENET:
                return false;
            case VGG16:
                return true;
            default:
                throw new IllegalStateException("Unknown modelWrapper type.");
        }
    }

}
