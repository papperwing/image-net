package cz.muni.fi.image.net.core.objects;

import java.util.Set;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class DataSample {

    private final String imageLocation;
    private final Set<Label> label;

    public DataSample(
            final String imageLocation,
            final Set<Label> label
    ) {
        this.imageLocation = imageLocation;
        this.label = label;
    }

    public String getImageLocation() {
        return imageLocation;
    }

    public Set<Label> getLabelSet() {
        return label;
    }

    @Override
    public String toString() {
        return getImageLocation() + getLabelSet().toString();
    }
}
