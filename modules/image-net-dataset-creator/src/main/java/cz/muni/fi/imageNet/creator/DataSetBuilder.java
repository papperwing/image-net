package cz.muni.fi.imageNet.creator;

import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.dataset.DataSetImpl;
import cz.muni.fi.imageNet.Pojo.Label;
import java.util.Collection;
import java.util.List;

/**
 * Interface is used for building Dataset from prepared data (downloaded images and labeled locations).
 * 
 * @author Jakub Peschel
 */
public interface DataSetBuilder {
    
    DataSetImpl buildDataSet(Collection<DataSample> downloadResult, List<Label> labels);

}
