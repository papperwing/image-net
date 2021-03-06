package cz.muni.fi.image.net.downloader.object;

import cz.muni.fi.image.net.downloader.enums.DownloadState;

import java.io.File;

/**
 * Class containt image and state of download of the image.
 *
 * @author Jakub Peschel
 */
public class DataImage {

    private final UrlImage urlImage;
    private File image;
    private DownloadState state;

    public DataImage(final UrlImage urlImage) {
        this.urlImage = urlImage;
    }

    //<editor-fold defaultstate="collapsed" desc="GET / SET">
    public UrlImage getUrlImage() {
        return urlImage;
    }

    public File getImage() {
        return image;
    }

    public void setImage(final File image) {
        this.image = image;
    }

    public DownloadState getState() {
        return state;
    }

    public void setState(final DownloadState state) {
        this.state = state;
    }
    //</editor-fold>

}
