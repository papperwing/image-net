/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.fi.image.net.api.dto;

import com.opencsv.CSVReader;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Jakub Peschel
 */
public class DataSampleDTOTest {

    /**
     * Test of getUrl method, of class DataSampleDTO.
     */
    @Test
    public void testCreateFromCSV() throws MalformedURLException, IOException {
        System.out.println("testCreateFromCSV");
        DataSampleDTO instance = new DataSampleDTO("\"http://test.test\",\"label\"");
        assertEquals(new URL("http://test.test"), instance.url);
        assertArrayEquals(new String[]{"label"}, instance.labels);
    }

    /**
     * Test of toCSVLine method, of class DataSampleDTO.
     */
    @Test
    public void testToCSVLine() throws IOException {
        System.out.println("toCSVLine");
        DataSampleDTO instance = new DataSampleDTO("\"http://test.test\",\"label\"", ',', ':');
        String expResult = "\"http://test.test\",\"label\"";
        String result = instance.toCSVLine();
        assertEquals(expResult, result);
    }

    @Test
    public void test() throws IOException {
        InputStream stream = new ByteArrayInputStream("\"url\",\"label1:label2\"".getBytes(StandardCharsets.UTF_8));
        CSVReader reader = new CSVReader(new InputStreamReader(stream, StandardCharsets.UTF_8), ',');
        String[] params = reader.readNext();
        assertArrayEquals(new String[]{"url", "label1:label2"}, params);
    }

}
