package cz.muni.fi.image.net.api;

import cz.muni.fi.image.net.api.dto.DataSampleDTO;
import cz.muni.fi.image.net.core.objects.Label;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.downloader.object.DataImage;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.downloader.object.DownloadResult;
import cz.muni.fi.image.net.downloader.object.UrlImage;
import cz.muni.fi.image.net.downloader.enums.DownloadState;
import cz.muni.fi.image.net.downloader.manager.ImageDownloadManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Class contains methods for processing of DataSampleDTO for other purpose.
 *
 * @author Jakub Peschel
 */
public class DataSampleTranslator {
    final Configuration config;

    /**
     * Constructor of {@link DataSampleTranslator}
     *
     * @param config global {@link Configuration}
     */
    public DataSampleTranslator(final Configuration config) {
        this.config = config;
    }

    /**
     * Translate {@link DataSampleDTO} into {@link DataSample}
     *
     * @param dataSamples array of {@link DataSampleDTO}
     * @return List of translated {@link DataSample}
     */
    public List<DataSample> getDataSampleCollection(final DataSampleDTO[] dataSamples) {
        final List<DataSample> result = new ArrayList();
        final List<UrlImage> urlImages = new ArrayList();
        final ImageDownloadManager downloadManager = new ImageDownloadManager(config);
        for (final DataSampleDTO dataSample : dataSamples) {
            urlImages.add(transformToUrlImage(dataSample));
        }
        result.addAll(transformToDataSamples(downloadManager.processRequest(urlImages)));
        return result;
    }

    /**
     * Retrieve {@link Label}s from array of {@link DataSampleDTO}
     *
     * @param dataSamples array of {@link DataSampleDTO}
     * @return {@link List} of unique {@link Label}s from dataSamples
     */
    public List<Label> getDataSampleLabels(final DataSampleDTO[] dataSamples) {
        final Set<Label> labels = new TreeSet<Label>();
        for (final DataSampleDTO dataSample : dataSamples) {
            for (final String labelName : dataSample.getLabels()) {
                labels.add(new Label(labelName));
            }
        }
        return new ArrayList(labels);
    }

    /**
     * Translate {@link DownloadResult} into {@link List} of {@link DataSample}s
     *
     * @param processRequest {@link DownloadResult}
     * @return {@link List} of {@link DataSample}s
     */
    public List<DataSample> transformToDataSamples(final DownloadResult processRequest) {
        final List<DataSample> result = new ArrayList<DataSample>();
        for (final DataImage dataImage : processRequest.getDataImageList()) {
            //TODO: solve problem with not downloaded data
            if (dataImage.getState() == DownloadState.DOWNLOADED
                    || dataImage.getState() == DownloadState.ON_DISK) {
                result.add(
                        new DataSample(
                                dataImage.getImage().getAbsolutePath(),
                                dataImage.getUrlImage().getLabels()
                        )
                );
            }
        }
        return result;
    }

    /**
     * Translate {@link DataSampleDTO} into {@link UrlImage}
     *
     * @param dataSample {@link DataSampleDTO}
     * @return {@link UrlImage}
     */
    public UrlImage transformToUrlImage(final DataSampleDTO dataSample) {
        final Set<Label> labels = new TreeSet();
        for (final String labelName : dataSample.getLabels()) {
            labels.add(new Label(labelName));
        }
        return new UrlImage(dataSample.getUrl(), labels);
    }
}
