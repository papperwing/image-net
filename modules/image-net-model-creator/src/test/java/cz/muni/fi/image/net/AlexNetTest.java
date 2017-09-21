package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import cz.muni.fi.image.net.model.creator.ModelBuilder;
import cz.muni.fi.image.net.model.creator.ModelBuilderImpl;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class AlexNetTest {

    private static final String TEMP_PATH = "./tmp";

    @Ignore
    @Test
    public void alexNetTest() {
        final Configuration config = new Configuration();
        config.setTempFolder(TEMP_PATH);
        config.setEpoch(10);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final NeuralNetModelWrapper model = modelBuilder.createModel(
                ModelType.ALEXNET,
                new DummyDataSet()
        );

        final TestTrainer trainer = new TestTrainer(config, model);

        final DataSetIterator testIterator = new ExistingMiniBatchDataSetIterator(
                new File(config.getTempFolder() + "/test/test"),
                "test-%d.bin"
        );

        final DataSetIterator trainIterator = new ExistingMiniBatchDataSetIterator(
                new File(config.getTempFolder() + "/test/train"),
                "train-%d.bin"
        );

        trainer.trainModel(
                testIterator,
                trainIterator,
                new DummyDataSet().getLabels()
        );
    }
}
