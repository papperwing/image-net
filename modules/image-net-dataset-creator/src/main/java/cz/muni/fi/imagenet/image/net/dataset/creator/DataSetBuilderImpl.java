package cz.muni.fi.imagenet.image.net.dataset.creator;

import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import java.util.Collection;
import java.util.List;

public class DataSetBuilderImpl implements DataSetBuilder {

    final Configuration config;

    public DataSetBuilderImpl(Configuration config) {
        this.config = config;
    }
    
    public DataSet buildDataSet(Collection<DataSample> dataSampleCollection, List<Label> labels) {
        return new DataSet(dataSampleCollection, labels);
    }

}
