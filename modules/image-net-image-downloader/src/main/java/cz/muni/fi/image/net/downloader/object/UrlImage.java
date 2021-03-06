package cz.muni.fi.image.net.downloader.object;

import cz.muni.fi.image.net.core.objects.Label;

import java.net.URL;
import java.util.Set;

/**
 * Class is supposed to contain information about image for downloading it.
 *
 * @author Jakub Peschel
 */
public class UrlImage {

    private final URL url;

    public Set<Label> getLabels() {
        return labels;
    }

    private final Set<Label> labels;

    public UrlImage(
            final URL url,
            final Set<Label> labels
    ) {
        this.url = url;
        this.labels = labels;
    }

    //<editor-fold defaultstate="collapsed" desc="GET / SET">
    public URL getUrl() {
        return url;
    }
    //</editor-fold>

}
