package cz.muni.fi.image.net.model.creator;

import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;

/**
 * Interface for training models and for clasification
 * @author Jakub Peschel
 */
public interface ModelBuilder {
    
    NeuralNetModel createModel(
            ModelType modelType, 
            DataSet dataSet
    );
    
}
