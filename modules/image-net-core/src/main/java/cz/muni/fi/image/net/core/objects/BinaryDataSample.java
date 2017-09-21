package cz.muni.fi.image.net.core.objects;

import java.util.Set;

/**
 *
 *  @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class BinaryDataSample {
    private final String imageLocation;
    private final boolean label;

    public BinaryDataSample(
            final String imageLocation,
            final boolean label
    ) {
        this.imageLocation = imageLocation;
        this.label = label;
    }

    public String getImageLocation() {
        return imageLocation;
    }

    public boolean isLabel() {
        return label;
    }

    @Override
    public String toString() {
        return getImageLocation() + "-"+ isLabel();
    }
}
