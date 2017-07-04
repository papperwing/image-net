package cz.muni.fi.image.net.downloader.object;

/**
 * Statistics of image batch download
 *
 * @author Jakub Peschel
 */
public class DownloadStatistics {

    /**
     * Number of downloaded images from batch.
     */
    private int downloaded = 0;
    /**
     * Number of images, which system wasnt able to download because of internet traffic.
     */
    private int failed = 0;
    /**
     * Number of images, which timed out during waiting for download.
     */
    private int timedOut = 0;
    /**
     * Number of images, which was allready on disk.
     */
    private int onDisk = 0;

    //<editor-fold defaultstate="collapsed" desc="GET / SET">
    public int getDownloaded() {
        return downloaded;
    }

    public void setDownloaded(int downloaded) {
        this.downloaded = downloaded;
    }

    public int getFailed() {
        return failed;
    }

    public void setFailed(int failed) {
        this.failed = failed;
    }

    public int getTimedOut() {
        return timedOut;
    }

    public void setTimedOut(int timedOut) {
        this.timedOut = timedOut;
    }

    public void addDownloaded() {
        this.downloaded++;
    }

    public void addFailed() {
        this.failed++;
    }

    public void addTimedOut() {
        this.timedOut++;
    }

    public int getOnDisk() {
        return onDisk;
    }

    public void setOnDisk(int onDisk) {
        this.onDisk = onDisk;
    }
    
    public void addOnDisk() {
        this.onDisk++;
    }
    //</editor-fold>

    @Override
    public String toString() {
        String msg = new StringBuilder()
                .append("|------------------")
                .append(System.lineSeparator())
                .append("|Succesfull: ")
                .append(getDownloaded() + getOnDisk())
                .append(System.lineSeparator())
                .append("|Downloaded: ")
                .append(getDownloaded())
                .append(System.lineSeparator())
                .append("|Failed: ")
                .append(getFailed() + getTimedOut())
                .append(System.lineSeparator())
                .append("|From Disk: ")
                .append(getDownloaded() + getOnDisk())
                .append(System.lineSeparator())
                .append("|------------------")
                .toString();
        return msg;
    }
    
    
}
