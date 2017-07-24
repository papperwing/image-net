package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.manager.ImageNetTrainer;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.NeuralNetModel;
import cz.muni.fi.image.net.model.creator.ModelBuilder;
import cz.muni.fi.image.net.model.creator.ModelBuilderImpl;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

        Configuration config = new Configuration();
        config.setTempFolder(TEMP_PATH);
        config.setEpoch(10);

        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);

        NeuralNetModel model = modelBuilder.createModel(
                ModelType.RESNET50,
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

class TestTrainer extends ImageNetTrainer {

    public TestTrainer(Configuration conf) {
        super(conf);
    }

    @Override
    public NeuralNetModel trainModel(
            NeuralNetModel model,
            DataSetIterator testIterator,
            DataSetIterator trainIterator,
            List<Label> labels
    ) {
        return super.trainModel(
                model,
                testIterator,
                trainIterator,
                labels
        );
    }
}

class DummyDataSet implements DataSet {

    public List<DataSample> getData() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public int lenght() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public List<Label> getLabels() {
        return Arrays.asList(
                new Label[]{
                        new Label("1"),
                        new Label("2"),
                        new Label("3"),
                        new Label("4"),
                        new Label("5"),
                        new Label("6"),
                        new Label("7")}
        );
    }

    public DataSet split(double percentage) {
        return new DummyDataSet();
    }

    public Map<Label, Integer> getLabelDistribution() {
        return new HashMap<>();
    }
}
