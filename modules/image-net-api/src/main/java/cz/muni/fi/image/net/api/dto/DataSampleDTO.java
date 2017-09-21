package cz.muni.fi.image.net.api.dto;

import com.opencsv.CSVReader;
import scala.Char;

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

    private static final char SEPARATOR = ',';
    private static final char LABEL_SEPARATOR = ':';

    private char separator;
    private char labelSeparator;
    final URL url;
    final String[] labels;

    /**
     * DataSample loaded from csv format line with {@link DataSampleDTO#SEPARATOR} and {@link DataSampleDTO#LABEL_SEPARATOR}
     *
     * @param csvLine {@link String}in format of csv
     * @throws IOException
     */
    public DataSampleDTO(String csvLine) throws IOException {
        this(csvLine, SEPARATOR, LABEL_SEPARATOR);
    }

    /**
     * Constructor of {@link DataSampleDTO}
     *
     * @param csvLine        {@link String}in format of csv
     * @param separator      columns separator
     * @param labelSeparator label names separator
     * @throws IOException
     */
    public DataSampleDTO(
            final String csvLine,
            final char separator,
            final char labelSeparator
    ) throws IOException {
        this.separator = separator;
        this.labelSeparator = labelSeparator;
        final InputStream stream = new ByteArrayInputStream(csvLine.getBytes(StandardCharsets.UTF_8));
        final CSVReader reader = new CSVReader(new InputStreamReader(stream, StandardCharsets.UTF_8), separator, '\"');
        final String[] params = reader.readNext();
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
        final StringBuilder message = new StringBuilder();
        message.append("URL: ").append(url.toString()).append(System.lineSeparator());
        message.append("Labels: ").append(labels).append(System.lineSeparator());
        return message.toString();
    }

    /**
     * Transform dataSample to cvs styled string
     *
     * @return String containing data in csv format
     */
    public String toCSVLine() {
        final StringBuilder csvLine = new StringBuilder()
                .append('"')
                .append(url)
                .append('"')
                .append(separator)
                .append('"');
        int index = 0;
        for (final String label : labels) {
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
