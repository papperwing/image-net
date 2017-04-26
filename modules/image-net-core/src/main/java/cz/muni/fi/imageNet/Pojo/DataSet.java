package cz.muni.fi.imageNet.Pojo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 * @author Jakub Peschel
 */
public class DataSet {

    private final Collection<DataSample> dataList;

    private final int dataSetLenght;

    private final List<Label> labels;

    public DataSet(final Collection<DataSample> dataList, List<Label> labels) {
        this.dataList = dataList;
        this.dataSetLenght = dataList.size();
        this.labels = labels;
        for (DataSample sample : dataList) {
            if (!labels.containsAll(sample.getLabelSet())) {
                throw new IllegalArgumentException("DataSample obsahuje neznámý label" + sample.toString());
            }
        }
    }

    public Collection<DataSample> getData() {
        return Collections.unmodifiableCollection(dataList);
    }

    public int getLenght() {
        return this.dataSetLenght;
    }

    public List<Label> getLabels() {
        return Collections.unmodifiableList(labels);
    }

    public List<String> getLabelStrings() {
        List<String> result = new ArrayList();
        for (Label label : labels) {
            result.add(label.getLabelName());
        }
        return Collections.unmodifiableList(new ArrayList(result));
    }

}
