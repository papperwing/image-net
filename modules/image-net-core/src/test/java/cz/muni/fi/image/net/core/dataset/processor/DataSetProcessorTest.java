package cz.muni.fi.image.net.core.dataset.processor;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class DataSetProcessorTest {

    @Before
    public void setUpDataSet() throws Exception{
        final File csv = new File(this.getClass().getClassLoader().getResource("testData.csv").getFile());
        InputStream stream = new FileInputStream(csv);
    }
    DataSet dataset;
    @Test
    public void prepareDataSetIterator() throws Exception {
        System.out.println("DataSetProcessorTest.prepareDataSetIterator()");
    }

    @Test
    public void presaveDataSetIterator() throws Exception {
        System.out.println("DataSetProcessorTest.presaveDataSetIterator()");
    }

}

class DummyDataSet implements DataSet{

    List<DataSample> data;
    List<Label> labels;

    @Override
    public List<DataSample> getData() {
        return data;
    }

    @Override
    public int lenght() {
        return data.size();
    }

    @Override
    public List<Label> getLabels() {
        return labels;
    }

    @Override
    public DataSet split(double percentage) {
        throw new UnsupportedOperationException("split() not enabled for test.");
    }

    @Override
    public Map<Label, Integer> getLabelDistribution() {
        throw new UnsupportedOperationException("getLabelDistribution() not enabled for test.");
    }
}