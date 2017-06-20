package cz.muni.fi.imageNet.dataset.creator;

import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.dataset.dataset.DataSetImpl;
import cz.muni.fi.imageNet.core.objects.Label;
import java.util.List;

public class DataSetBuilderImpl implements DataSetBuilder {

    final Configuration config;

    public DataSetBuilderImpl(Configuration config) {
        this.config = config;
    }
    
    public DataSetImpl buildDataSet(List<DataSample> dataSampleCollection, List<Label> labels) {
        return new DataSetImpl(dataSampleCollection, labels);
    }

}
