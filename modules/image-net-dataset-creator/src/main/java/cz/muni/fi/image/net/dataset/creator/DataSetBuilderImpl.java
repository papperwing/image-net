package cz.muni.fi.image.net.dataset.creator;

import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.dataset.dataset.DataSetImpl;

import java.util.List;

public class DataSetBuilderImpl implements DataSetBuilder {

    final Configuration config;

    /**
     * Constructor of {@link DataSetBuilderImpl}
     *
     * @param config global {@link Configuration}
     */
    public DataSetBuilderImpl(Configuration config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    public DataSetImpl buildDataSet(List<DataSample> dataSampleCollection, List<Label> labels) {
        return new DataSetImpl(dataSampleCollection, labels);
    }

    /**
     * {@inheritDoc}
     */
    public DataSetImpl buildBinaryDataSet(List<DataSample> dataSampleCollection, List<Label> labels) {
        return new DataSetImpl(dataSampleCollection, labels);
    }

}
