package cz.muni.fi.image.net.core.manager;

import cz.muni.fi.image.net.core.objects.Label;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class LabelHelper {

    public static List<String> translate(final List<Label> labels) {
        final List<String> result = new ArrayList();
        for (final Label label : labels) {
            result.add(label.getLabelName());
        }
        return result;
    }

}
