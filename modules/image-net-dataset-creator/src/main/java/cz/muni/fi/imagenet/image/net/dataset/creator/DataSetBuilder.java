package cz.muni.fi.imagenet.image.net.dataset.creator;

import cz.muni.fi.imageNet.Pojo.DataSample;
import cz.muni.fi.imageNet.Pojo.DataSet;
import cz.muni.fi.imageNet.Pojo.Label;
import java.util.Collection;
import java.util.List;

/**
 * Interface is used for building Dataset from prepared data (downloaded images and labeled locations).
 * 
 * @author Jakub Peschel
 */
public interface DataSetBuilder {
    
    DataSet buildDataSet(Collection<DataSample> downloadResult, List<Label> labels);

}
