package cz.muni.fi.imageNet;

import cz.muni.fi.imageNet.core.manager.ImageNetRunner;
import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.Label;
import cz.muni.fi.imageNet.core.objects.ModelType;
import cz.muni.fi.imageNet.core.objects.NeuralNetModel;
import cz.muni.fi.imageNet.model.creator.ModelBuilder;
import cz.muni.fi.imageNet.model.creator.ModelBuilderImpl;
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
 *
 * @author jpeschel
 */
public class ResnetMemoryConsumptionTest {

//    @Ignore
    @Test
    public void ResnetMemoryConsumptionTest() {

        Configuration config = new Configuration();
        config.setTempFolder("/home/xpeschel/tmp");
        config.setEpoch(1);
        final ModelBuilder modelBuilder = new ModelBuilderImpl(config);
        NeuralNetModel model = modelBuilder.createModel(ModelType.RESNET50, new DummyDataSet());
        final ImageNetRunner runner = new ImageNetRunner(config){

            @Override
            protected DataSetIterator prepareDataSetIterator(DataSet dataset, ModelType modelType, String saveDataName) {
                final File saveFolder = new File(this.conf.getTempFolder() + File.separator + "minibatches" + File.separator + saveDataName);
		System.out.println(saveFolder.getAbsolutePath());
                return new ExistingMiniBatchDataSetIterator(saveFolder, saveDataName + "-%d.bin");
            }
            
        };
        runner.trainModel(model, new DummyDataSet());
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
