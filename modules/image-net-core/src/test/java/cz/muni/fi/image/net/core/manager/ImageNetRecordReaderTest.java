package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.data.sample.processing.ImageNetRecordReader;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

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

    ImageNetRecordReader instance = new ImageNetRecordReader(0, 0, 0, ModelType.RESNET50);

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
        try {
            instance.initialize(dataset, null);
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
        public List<DataSample> getData() {
            return sampleList;
        }

        @Override
        public int length() {
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
