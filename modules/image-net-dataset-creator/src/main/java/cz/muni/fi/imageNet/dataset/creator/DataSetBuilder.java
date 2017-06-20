package cz.muni.fi.imageNet.dataset.creator;

import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.dataset.dataset.DataSetImpl;
import cz.muni.fi.imageNet.core.objects.Label;
import java.util.Collection;
import java.util.List;

/**
 * Interface is used for building Dataset from prepared data (downloaded images and labeled locations).
 * 
 * @author Jakub Peschel
 */
public interface DataSetBuilder {
    
    DataSetImpl buildDataSet(List<DataSample> downloadResult, List<Label> labels);

}
