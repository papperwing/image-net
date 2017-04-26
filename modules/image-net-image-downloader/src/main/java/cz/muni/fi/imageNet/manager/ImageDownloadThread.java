package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.DataImage;
import cz.muni.fi.imageNet.Pojo.DownloadResult;
import cz.muni.fi.imageNet.Pojo.UrlImage;
import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.enums.DownloadState;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Worker for managing download of one image
 *
 * @author Jakub Peschel
 */
public class ImageDownloadThread implements Runnable {

    private final Configuration conf;

    private final Logger logger = LoggerFactory.getLogger(ImageDownloadThread.class);
    /**
     * Container used for storing {@link DataImage}
     */
    private final DownloadResult downloadResult;
    /**
     * Url and info for image download
     */
    private final UrlImage urlImage;

    /**
     * Constructor for image download worker.
     *
     * @param urlImage image to download
     * @param downloadResult result where to store {@link DataImage}
     */
    public ImageDownloadThread(UrlImage urlImage, final DownloadResult downloadResult, Configuration conf) {
        this.downloadResult = downloadResult;
        this.urlImage = urlImage;
        this.conf = conf;
    }

    /**
     * Method download file and save it into location defined by {@link Configuration#imageDownloadFolder}.
     */
    public void run() {
        /*
         * Download image into defined disk folder
         */
        URL url = urlImage.getUrl();

        DataImage result = new DataImage(urlImage);

        final String path = conf.getImageDownloadFolder() + File.separator + url.getFile();
        File imageFile = new File(path);
        imageFile.getParentFile().mkdirs();

        if (!imageFile.isFile()) {
            try {
                downloadUsingNIO(url.toString(), path);
            } catch (IOException ex) {
                logger.error("Downloading of file " + urlImage.getUrl().toString() + " failed.", ex);
                result.setState(DownloadState.FAILED);
                writeAndNotify(result);
                return;
            }
            result.setState(DownloadState.DOWNLOADED);

        } else {
            result.setState(DownloadState.ON_DISK);
        }
        result.setImage(imageFile);
        writeAndNotify(result);
    }

    private void writeAndNotify(DataImage result) {
        synchronized (downloadResult) {
            downloadResult.getDataImageList().add(result);
            downloadResult.notifyAll();
        }
    }

    private void downloadUsingNIO(String urlStr, String file) throws IOException {
        URL url = new URL(urlStr);
        try (ReadableByteChannel rbc = Channels.newChannel(url.openStream())) {
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
            }
        }
    }

}
