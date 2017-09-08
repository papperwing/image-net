package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.manager.ImageNetTrainer;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

public class TestTrainer extends ImageNetTrainer {


    public TestTrainer(
            Configuration conf,
            NeuralNetModelWrapper model
    ) {
        super(conf, model);
    }

    @Override
    public NeuralNetModelWrapper trainModel(
            DataSetIterator testIterator,
            DataSetIterator trainIterator,
            List<Label> labels
    ) {
        return super.trainModel(
                testIterator,
                trainIterator,
                labels
        );
    }
}
