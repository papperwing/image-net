package cz.muni.fi.imageNet.dataset;

import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author Jakub Peschel
 */
public class DataSetImpl implements DataSet {

    private Collection<DataSample> dataList;

    private final int dataSetLenght;

    private final List<Label> labels;

    public DataSetImpl(final Collection<DataSample> dataList, List<Label> labels) {
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

    public int lenght() {
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
    
    public DataSet split(double splitPercentage){
        if(splitPercentage < 0 || splitPercentage > 1) throw new IllegalArgumentException("Percentage must be between 0 and 1.");
        int splitIndex = (int) Math.ceil(this.dataSetLenght*splitPercentage); 
        
        Collection<DataSample> splitList = new ArrayList(this.dataList).subList(splitIndex, dataSetLenght-1);
        dataList = new ArrayList(this.dataList).subList(0, splitIndex);
        DataSetImpl splitSet = new DataSetImpl(splitList, labels);
    
        return splitSet;
    }

}
