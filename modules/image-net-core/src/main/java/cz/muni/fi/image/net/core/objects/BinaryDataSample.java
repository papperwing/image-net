package cz.muni.fi.image.net.core.objects;

import java.util.Set;

public class BinaryDataSample {
    private final String imageLocation;
    private final boolean label;

    public BinaryDataSample(String imageLocation, boolean label) {
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
