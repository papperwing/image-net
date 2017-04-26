package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.ModelType;
import cz.muni.fi.imageNet.Pojo.NeuralNetModel;
import cz.muni.fi.imageNet.Pojo.NetworkConfiguration;

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
