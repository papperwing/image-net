package cz.muni.fi.image.net.downloader.manager;

import cz.muni.fi.image.net.downloader.object.DownloadResult;
import cz.muni.fi.image.net.downloader.object.DataImage;
import cz.muni.fi.image.net.downloader.object.UrlImage;
import cz.muni.fi.image.net.core.objects.Configuration;

import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manager for downloading images.
 *
 * @author Jakub Peschel
 */
public class ImageDownloadManager {

    private final Logger logger = LoggerFactory.getLogger(ImageDownloadManager.class);
    private final Configuration conf;

    private final ThreadPoolExecutor pool;

    /**
     * Constructor of {@link ImageDownloadManager}
     *
     * @param conf global {@link Configuration}
     */
    public ImageDownloadManager(final Configuration conf) {
        this.conf = conf;
        this.pool = new ThreadPoolExecutor(
                conf.getCorePoolSize(),
                conf.getMaximumPoolSize(),
                conf.getKeepAliveTime(),
                TimeUnit.DAYS,
                new LinkedBlockingQueue<Runnable>()
        );

    }

    /**
     * Downloading images based on request list
     *
     * @param requestList {@link List} of {@link UrlImage}
     * @return {@link DownloadResult}
     */
    public DownloadResult processRequest(final List<UrlImage> requestList) {
        final DownloadResult result = new DownloadResult();
        for (final UrlImage image : requestList) {
            final ImageDownloadThread thread = new ImageDownloadThread(image, result, conf);
            pool.execute(thread);
        }

        // waiting till all download threads are not processed
        while (result.getDataImageList().size() != requestList.size()) {
            logger.info(
                    "Resolved files: "
                            + result.getDataImageList().size()
            );
            try {
                synchronized (result) {
                    result.wait(10000);
                }
            } catch (InterruptedException ex) {
                logger.error("Wait was interrupted", ex);
            }
        }
        writeStatistics(result);
        logger.info("Statistics: \n" + result.getStats().toString());
        pool.shutdown();
        return result;
    }

    private void writeStatistics(final DownloadResult result) {
        for (final DataImage image : result.getDataImageList()) {
            switch (image.getState()) {
                case DOWNLOADED: {
                    result.getStats().addDownloaded();
                    break;
                }

                case FAILED: {
                    result.getStats().addFailed();
                    break;
                }

                case TIMED_OUT: {
                    result.getStats().addTimedOut();
                    break;
                }

                case ON_DISK: {
                    result.getStats().addOnDisk();
                    break;
                }
            }
        }
    }
}
