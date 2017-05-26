package cz.muni.fi.imageNet.downloader.object;

import java.util.Collection;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Collection of downloadeed images with status and statistics of download
 *
 * @author Jakub Peschel
 */
public class DownloadResult {

    private final Collection<DataImage> dataImageList = new ConcurrentLinkedQueue<DataImage>();

    private final DownloadStatistics stats = new DownloadStatistics();

    //<editor-fold defaultstate="collapsed" desc="GET / SET">
    public Collection<DataImage> getDataImageList() {
        return dataImageList;
    }

    public DownloadStatistics getStats() {
        return stats;
    }
    //</editor-fold>

}
