package cz.muni.fi.image.net.dataset.creator;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.dataset.dataset.DataSetImpl;

import java.util.List;

/**
 * Interface is used for building {@link DataSet} from prepared data (downloaded images and labeled locations).
 *
 * @author Jakub Peschel
 */
public interface DataSetBuilder {

    /**
     * Create {@link DataSet} from {@link DataSample}s
     *
     * @param downloadResult {@link List} of {@link DataSample}s
     * @param labels         {@link List} of {@link Label}s
     * @return {@link DataSet}
     */
    DataSet buildDataSet(List<DataSample> downloadResult, List<Label> labels);

}
