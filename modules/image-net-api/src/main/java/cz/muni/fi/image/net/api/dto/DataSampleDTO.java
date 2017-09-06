package cz.muni.fi.image.net.api.dto;

import com.opencsv.CSVReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

/**
 * DTO for dataset data
 *
 * @author Jakub Peschel (jakub.peschel@studentagency.cz)
 */
public class DataSampleDTO {

    private char separator;
    private char labelSeparator;
    final URL url;
    final String[] labels;

    /**
     *
     *
     * @param csvLine
     * @throws MalformedURLException
     * @throws IOException
     */
    public DataSampleDTO(String csvLine) throws MalformedURLException, IOException {
        this(csvLine, ',', ':');
    }

    /**
     *
     * @param csvLine
     * @param separator
     * @param labelSeparator
     * @throws MalformedURLException
     * @throws IOException
     */
    public DataSampleDTO(String csvLine, char separator, char labelSeparator) throws MalformedURLException, IOException {
        this.separator = separator;
        this.labelSeparator = labelSeparator;
        InputStream stream = new ByteArrayInputStream(csvLine.getBytes(StandardCharsets.UTF_8));
        CSVReader reader = new CSVReader(new InputStreamReader(stream, StandardCharsets.UTF_8), separator, '\"');
        String[] params = reader.readNext();
        if (params.length < 2) {
            throw new IllegalArgumentException("csvLine has missing parameters");
        }
        if (params.length > 2) {
            throw new IllegalArgumentException("csvLine has to much parameters");
        }

        this.url = new URL(params[0]);
        this.labels = params[1].split(String.valueOf(labelSeparator));
    }

    public DataSampleDTO(URL url, String[] labels) {
        this.labels = labels;

        this.url = url;

    }

    public URL getUrl() {
        return url;
    }

    public String[] getLabels() {
        return labels;
    }

    public List<String> getLabelList() {
        return Arrays.asList(labels);
    }

    @Override
    public String toString() {
        StringBuilder message = new StringBuilder();
        message.append("URL: ").append(url.toString()).append(System.lineSeparator());
        message.append("Labels: ").append(labels).append(System.lineSeparator());
        return message.toString();
    }

    public String toCSVLine() {
        final StringBuilder csvLine = new StringBuilder()
                .append('"')
                .append(url)
                .append('"')
                .append(separator)
                .append('"');
        int index = 0;
        for (String label : labels) {
            if (index != 0) {
                csvLine.append(labelSeparator);
            }
            csvLine.append(label);
            index++;
        }
        csvLine.append('"');
        return csvLine.toString();
    }
}
