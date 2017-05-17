package cz.muni.fi.imageNet.Pojo;

import java.util.Collection;
import java.util.List;

/**
 *
 * @author jpeschel
 */
public interface DataSet {

    Collection<DataSample> getData();
    
    int lenght();
    
    List<Label> getLabels();
    
    
    DataSet split(double percentage);
    
}
