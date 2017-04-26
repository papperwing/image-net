package cz.muni.fi.imageNet.POJO;

import java.net.MalformedURLException;
import java.net.URL;

/**
 * DTO for data
 *
 * @author Jakub Peschel (jakub.peschel@studentagency.cz)
 */
public class DataSampleDTO {

    public static String SEPARATOR = ";";
    public static String LABEL_SEPARATOR = ",";
    final URL url;
    final String[] labels;

    public DataSampleDTO(String csvLine) throws MalformedURLException {
        String[] params = csvLine.split(SEPARATOR);
        if (params.length < 2) {
            throw new IllegalArgumentException("csvLine has missing parameters");
        }
        if (params.length > 2) {
            throw new IllegalArgumentException("csvLine has to much parameters");
        }

        this.url = new URL(params[0]);
        this.labels = params[1].split(LABEL_SEPARATOR);
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

    @Override
    public String toString() {
        StringBuilder message = new StringBuilder();
        message.append("URL: ").append(url.toString()).append(System.lineSeparator());
        message.append("Labels: ").append(labels).append(System.lineSeparator());
        return message.toString();
    }

    public String toCSVLine() {
        final StringBuilder csvLine = new StringBuilder()
                .append(url)
                .append(SEPARATOR);
        int index = 0;
        for (String label : labels) {
            if (index != 0) {
                csvLine.append(",");
            }
            csvLine.append(label);
            index++;
        }
        return csvLine.toString();
    }

}
