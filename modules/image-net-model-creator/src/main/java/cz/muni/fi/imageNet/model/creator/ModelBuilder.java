package cz.muni.fi.imageNet.model.creator;

import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.ModelType;
import cz.muni.fi.imageNet.core.objects.NeuralNetModel;

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
