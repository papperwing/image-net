package cz.muni.fi.image.net;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DummyDataSet implements DataSet {

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
