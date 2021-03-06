package cz.muni.fi.image.net.core.dataset.processor;

import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import org.junit.Before;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.*;
import java.util.*;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class DataSetTransformTest {

    @Before
    public void setUpDataSet() throws Exception {
        final File csv = new File(this.getClass().getClassLoader().getResource("testData.csv").getFile());
        final File imagesDir = new File(this.getClass().getClassLoader().getResource("images").getFile());
        final InputStreamReader reader = new FileReader(csv);
        final BufferedReader bfReader = new BufferedReader(reader);
        String line;
        final List<DataSample> samples = new ArrayList<>();
        final Set<Label> labels = new HashSet<>();
        while ((line = bfReader.readLine()) != null) {
            final String[] splitted = line.split(";");
            final Set<Label> labelSet = new HashSet<>();
            for (final String labelstring : splitted[1].split(",")) {
                labelSet.add(new Label(labelstring));
            }
            final DataSample sample = new DataSample(
                    imagesDir.getAbsolutePath() + File.separator + splitted[0],
                    labelSet
            );
            samples.add(sample);
            labels.addAll(labelSet);
        }
        dataset = new DummyDataSet();
        ((DummyDataSet) dataset).data = samples;
        ((DummyDataSet) dataset).labels = new ArrayList<>(labels);
        config = new Configuration();
        config.setBatchSize(5);
        config.setImageDownloadFolder(imagesDir.getAbsolutePath());
    }

    private DataSet dataset;
    Configuration config;

    @Test
    public void prepareDataSetIterator() throws Exception {
        System.out.println("DataSetTransformTest.prepareDataSetIterator()");

        final DataSetTransform processor = new DataSetTransform(config, ModelType.ALEXNET);
        final DataSetIterator iter = processor.prepareDataSetIterator(dataset);
        while (iter.hasNext()){
            org.nd4j.linalg.dataset.DataSet testSet = iter.next();
            System.out.println();
            assert (testSet.getFeatures().minNumber().doubleValue() >= -1);
            assert (testSet.getFeatures().minNumber().doubleValue() <= 0);
            assert (testSet.getFeatures().maxNumber().doubleValue() >= 0);
            assert (testSet.getFeatures().maxNumber().doubleValue() <= 1);
        }
    }

    @Test
    public void presaveDataSetIterator() throws Exception {
        System.out.println("DataSetTransformTest.presaveDataSetIterator()");

        final TemporaryFolder temporaryFolder = new TemporaryFolder();
        temporaryFolder.create();
        final File temp = temporaryFolder.getRoot();

        config.setTempFolder(temp.getAbsolutePath());
        final DataSetTransform transform = new DataSetTransform(config, ModelType.ALEXNET);
        final DataSetIterator iter = transform.presaveDataSetIterator(
                transform.prepareDataSetIterator(dataset),
                ModelType.ALEXNET,
                "test"
        );
        while (iter.hasNext()){
            org.nd4j.linalg.dataset.DataSet testSet = iter.next();
            System.out.println();
            assert (testSet.getFeatures().minNumber().doubleValue() >= -1);
            assert (testSet.getFeatures().minNumber().doubleValue() <= 0);
            assert (testSet.getFeatures().maxNumber().doubleValue() >= 0);
            assert (testSet.getFeatures().maxNumber().doubleValue() <= 1);
        }
    }

}

class DummyDataSet implements DataSet {

    public List<DataSample> data;
    public List<Label> labels;

    @Override
    public List<DataSample> getData() {
        return data;
    }

    @Override
    public int length() {
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

    @Override
    public String toString() {
        return "[" + data.toString() + "],[" + labels.toString() + "]";
    }
}