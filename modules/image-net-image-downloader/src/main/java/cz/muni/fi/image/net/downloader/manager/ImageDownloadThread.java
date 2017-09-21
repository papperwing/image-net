package cz.muni.fi.image.net.downloader.manager;

import cz.muni.fi.image.net.downloader.object.DownloadResult;
import cz.muni.fi.image.net.downloader.object.DataImage;
import cz.muni.fi.image.net.downloader.object.UrlImage;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.downloader.enums.DownloadState;

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
     * Constructor for {@link ImageDownloadThread}.
     *
     * @param urlImage       image to download
     * @param downloadResult result where to store {@link DataImage}
     */
    public ImageDownloadThread(
            final UrlImage urlImage,
            final DownloadResult downloadResult,
            final Configuration conf
    ) {
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
        final URL url = urlImage.getUrl();

        final DataImage result = new DataImage(urlImage);

        final String path = conf.getImageDownloadFolder() + File.separator + url.getFile();
        final File imageFile = new File(path);
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

    private void writeAndNotify(final DataImage result) {
        synchronized (downloadResult) {
            downloadResult.getDataImageList().add(result);
            downloadResult.notifyAll();
        }
    }

    private void downloadUsingNIO(
            final String urlStr,
            final String file
    ) throws IOException {
        final URL url = new URL(urlStr);
        try (final ReadableByteChannel rbc = Channels.newChannel(url.openStream())) {
            try (final FileOutputStream fos = new FileOutputStream(file)) {
                fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
            }
        }
    }

}
