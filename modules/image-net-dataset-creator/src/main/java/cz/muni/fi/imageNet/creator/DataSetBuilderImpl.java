package cz.muni.fi.imageNet.creator;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.dataset.DataSetImpl;
import cz.muni.fi.imageNet.Pojo.Label;
import java.util.Collection;
import java.util.List;

public class DataSetBuilderImpl implements DataSetBuilder {

    final Configuration config;

    public DataSetBuilderImpl(Configuration config) {
        this.config = config;
    }
    
    public DataSetImpl buildDataSet(Collection<DataSample> dataSampleCollection, List<Label> labels) {
        return new DataSetImpl(dataSampleCollection, labels);
    }

}
