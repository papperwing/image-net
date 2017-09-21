package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.manager.ImageNetTrainer;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

public class TestTrainer extends ImageNetTrainer {


    public TestTrainer(
            final Configuration conf,
            final NeuralNetModelWrapper model
    ) {
        super(conf, model);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NeuralNetModelWrapper trainModel(
            final DataSetIterator testIterator,
            final DataSetIterator trainIterator,
            final List<Label> labels
    ) {
        return super.trainModel(
                testIterator,
                trainIterator,
                labels
        );
    }
}
