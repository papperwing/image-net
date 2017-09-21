package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModelWrapper;
import cz.muni.fi.image.net.model.creator.ModelBuilder;
import cz.muni.fi.image.net.model.creator.ModelBuilderImpl;

import java.io.File;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ResnetMemoryConsumptionTest {

    private static final String TEMP_PATH = "./tmp";

    @Ignore("Test need to be generalized, once I find limits etc.")
    @Test
    public void ResnetMemoryConsumptionTest() {

        final Configuration config = new Configuration();
        config.setTempFolder(TEMP_PATH);
        config.setEpoch(10);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        final NeuralNetModelWrapper model = modelBuilder.createModel(
                ModelType.RESNET50,
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



