/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
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

    /**
     * Test of getLabels method, of class ImageNetRecordReader.
     */
    @Test
    public void testGetLabels() {
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
        
        DataSet dataset = new DataSet(sampleList,labels);

        ImageNetSplit split = new ImageNetSplit(dataset);
        split.initialize();
        ImageNetRecordReader instance = new ImageNetRecordReader(0, 0, 0);
        try {
            instance.initialize(split);
            List<String> result = instance.getLabels();
            assertEquals(expResult, result);
        } catch (Exception ex) {
            fail("Test failed on Exception");
        }
    }

}
