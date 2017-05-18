/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.datavec.image.loader.NativeImageLoader;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.InjectMocks;
import static org.mockito.Matchers.any;
import org.mockito.Mock;
import org.mockito.Mockito;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.when;
import org.mockito.MockitoAnnotations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author jpeschel
 */
public class ImageNetRecordReaderTest {

    public ImageNetRecordReaderTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    ImageNetRecordReader instance = new ImageNetRecordReader(0, 0, 0);

    /**
     * Test of getLabels method, of class ImageNetRecordReader.
     */
    @Test
    public void testGetLabels() throws Exception {

        System.out.println("getLabels");
        List<Label> labels = new ArrayList();
        Set<Label> labels1 = new HashSet();
        final Label test1 = new Label("test1");
        labels1.add(test1);
        labels.add(test1);
        final Label test2 = new Label("test2");
        labels1.add(test2);
        labels.add(test2);
        DataSample sample1 = new DataSample("testLocation", labels1);

        Set<Label> labels2 = new HashSet();
        final Label test3 = new Label("test3");
        labels2.add(test3);
        labels.add(test3);
        final Label test4 = new Label("test4");
        labels2.add(test4);
        labels.add(test4);

        List<String> expResult = new ArrayList<String>();
        expResult.add("test1");
        expResult.add("test2");
        expResult.add("test3");
        expResult.add("test4");
        
        DataSample sample2 = new DataSample("testLocation2", labels2);
        final ArrayList sampleList = new ArrayList();
        sampleList.add(sample1);
        sampleList.add(sample2);

        DataSet dataset = new DataSetDummy(sampleList, labels);
        ImageNetSplit split = new ImageNetSplit(dataset);
        split.initialize();
        try {
            instance.initialize(split);
            List<String> result = instance.getLabels();
            assertEquals(expResult, result);
        } catch (Exception ex) {
            fail("Test failed on Exception: " + ex.toString());
        }
    }

    private static class DataSetDummy implements DataSet {

        List<DataSample> sampleList;
        List<Label> labels;

        public DataSetDummy(ArrayList sampleList, List<Label> labels) {
            this.sampleList = sampleList;
            this.labels = labels;
        }

        @Override
        public Collection<DataSample> getData() {
            return sampleList;
        }

        @Override
        public int lenght() {
            return sampleList.size();
        }

        @Override
        public List<Label> getLabels() {
            return labels;
        }

        @Override
        public DataSet split(double percentage) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Map<Label, Integer> getLabelDistribution() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

}
