package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.objects.Label;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author jpeschel
 */
public class LabelHelper {
    
    public static List<String> translate (List<Label> labels){
        List<String> result = new ArrayList();
        for (Label label : labels){
            result.add(label.getLabelName());
        }
        return result;
    }
    
}