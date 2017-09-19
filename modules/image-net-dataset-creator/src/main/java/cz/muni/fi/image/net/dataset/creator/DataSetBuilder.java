package cz.muni.fi.image.net.dataset.creator;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.dataset.dataset.DataSetImpl;

import java.util.List;

/**
 * Interface is used for building Dataset from prepared data (downloaded images and labeled locations).
 * 
 * @author Jakub Peschel
 */
public interface DataSetBuilder {
    
    DataSetImpl buildDataSet(List<DataSample> downloadResult, List<Label> labels);

    DataSetImpl buildBinaryDataSet(List<DataSample> downloadResult, List<Label> labels);

}
