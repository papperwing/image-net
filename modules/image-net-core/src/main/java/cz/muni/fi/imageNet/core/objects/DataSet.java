package cz.muni.fi.imageNet.core.objects;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 *
 * @author jpeschel
 */
public interface DataSet {

    Collection<DataSample> getData();
    
    int lenght();
    
    List<Label> getLabels();
    
    
    DataSet split(double percentage);
    

    Map<Label, Integer> getLabelDistribution();
    
}
