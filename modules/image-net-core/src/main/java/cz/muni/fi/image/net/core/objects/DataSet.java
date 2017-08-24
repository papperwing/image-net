package cz.muni.fi.image.net.core.objects;

import java.util.List;
import java.util.Map;

/**
 *
 * @author jpeschel
 */
public interface DataSet {

    List<DataSample> getData();
    
    int length();
    
    List<Label> getLabels();
    
    
    DataSet split(double percentage);
    

    Map<Label, Integer> getLabelDistribution();
    
}
