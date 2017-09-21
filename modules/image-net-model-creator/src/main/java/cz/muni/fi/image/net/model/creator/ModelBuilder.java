package cz.muni.fi.image.net.model.creator;

import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;

/**
 * Interface for training models and for clasification
 *
 * @author Jakub Peschel
 */
public interface ModelBuilder {

    /**
     * Create {@link NeuralNetModelWrapper} with defined {@link ModelType} based on statistics of {@link DataSet}
     *
     * @param modelType specified {@link ModelType} of underlying network architecture
     * @param dataSet   {@link DataSet}
     * @return pretrianed model stored in {@link NeuralNetModelWrapper}
     */
    NeuralNetModelWrapper createModel(
            ModelType modelType,
            DataSet dataSet
    );

}
