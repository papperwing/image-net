package cz.muni.fi.image.net.dataset.dataset;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Jakub Peschel
 */
public class DataSetImpl implements DataSet {

    private List<DataSample> dataList;

    private final int dataSetLength;

    private final List<Label> labels;

    public DataSetImpl(final List<DataSample> dataList, List<Label> labels) {
        this.dataList = dataList;
        this.dataSetLength = dataList.size();
        this.labels = labels;
        for (DataSample sample : dataList) {
            if (!labels.containsAll(sample.getLabelSet())) {
                throw new IllegalArgumentException("DataSample obsahuje neznámý label" + sample.toString());
            }
        }
    }

    public List<DataSample> getData() {
        return Collections.unmodifiableList(dataList);
    }

    public int length() {
        return this.dataSetLength;
    }

    public List<Label> getLabels() {
        return Collections.unmodifiableList(labels);
    }

    public List<String> getLabelStrings() {
        List<String> result = new ArrayList();
        for (Label label : labels) {
            result.add(label.getLabelName());
        }
        return Collections.unmodifiableList(result);
    }

    public DataSet split(double splitPercentage) {
        validatePecentage(splitPercentage);
        int splitIndex = (int) Math.ceil(this.dataSetLength * splitPercentage);

        List<DataSample> splitList = this.dataList.subList(splitIndex, dataSetLength - 1);
        dataList = this.dataList.subList(0, splitIndex);
        DataSetImpl splitSet = new DataSetImpl(splitList, labels);

        return splitSet;
    }

    public Map<Label, Integer> getLabelDistribution() {
        Map<Label, Integer> result = new HashMap();
        for (Label label : getLabels()) {
            result.put(label, new Integer(0));
        }

        for (DataSample sample : getData()) {
            for (Label label : sample.getLabelSet()) {
                result.put(label, result.get(label) + 1);
            }
        }
        
        return result;
    }  
    
    private void validatePecentage(double splitPercentage) throws IllegalArgumentException {
        if (splitPercentage < 0 || splitPercentage > 1) {
            throw new IllegalArgumentException("Percentage must be between 0 and 1.");
        }
    }


}
