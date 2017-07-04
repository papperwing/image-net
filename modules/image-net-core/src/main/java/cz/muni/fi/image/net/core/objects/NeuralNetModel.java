package cz.muni.fi.image.net.core.objects;

import java.io.File;
import java.io.IOException;
import java.util.List;

import cz.muni.fi.image.net.core.enums.ModelType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;

/**
 * Class for storing model of neural net
 *
 * @author Jakub Peschel
 */
public class NeuralNetModel {

    private final Model model;

    private final List<Label> labels;
    
    private final ModelType type;

    private String description;

    private boolean trained;
    

    public NeuralNetModel(final Model model, List<Label> labelList, ModelType type) {
        this.trained = false;
        this.labels = labelList;
        this.model = model;
        this.type = type;
    }

    public NeuralNetModel(final File savedModel, List<Label> labelList, ModelType type) throws IOException {
        this.trained = false;
        this.model = ModelSerializer.restoreComputationGraph(savedModel);
        this.labels = labelList;
        this.type = type;
    }

    public Model getModel() {
        return model;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public List<Label> getLabels() {
        return labels;
    }

    public File toFile(String storeFileLocation) throws IOException {
        String fileNameWithExtension = storeFileLocation + ".zip";
        File saveFile = new File(fileNameWithExtension);
        ModelSerializer.writeModel(model, saveFile, trained);//nechceme ukládat nenaučený model prozatím
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
    
}
