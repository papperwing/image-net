package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.manager.ImageNetTrainer;
import cz.muni.fi.image.net.core.objects.*;
import cz.muni.fi.image.net.model.creator.ModelBuilder;
import cz.muni.fi.image.net.model.creator.ModelBuilderImpl;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LeNetTest {

    private static final String TEMP_PATH = "./tmp";

    @Ignore("Test need to be generalized, once I find limits etc.")
    @Test
    public void leNetTest() {
        Configuration config = new Configuration();
        config.setTempFolder(TEMP_PATH);
        config.setEpoch(10);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        NeuralNetModel model = modelBuilder.createModel(
                ModelType.LENET,
                new DummyDataSet()
        );

        final TestTrainer trainer = new TestTrainer(config);

        DataSetIterator testIterator = new ExistingMiniBatchDataSetIterator(
                new File(config.getTempFolder() + "/test/test"),
                "test-%d.bin"
        );

        DataSetIterator trainIterator = new ExistingMiniBatchDataSetIterator(
                new File(config.getTempFolder() + "/test/train"),
                "train-%d.bin"
        );

        trainer.trainModel(
                model,
                testIterator,
                trainIterator,
                new DummyDataSet().getLabels()
        );


    }
}

