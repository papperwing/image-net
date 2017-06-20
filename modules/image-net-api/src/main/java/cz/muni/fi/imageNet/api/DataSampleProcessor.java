package cz.muni.fi.imageNet.api;

import cz.muni.fi.imageNet.core.objects.Configuration;
import cz.muni.fi.imageNet.downloader.object.DataImage;
import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.downloader.object.DownloadResult;
import cz.muni.fi.imageNet.core.objects.Label;
import cz.muni.fi.imageNet.downloader.object.UrlImage;
import cz.muni.fi.imageNet.api.dto.DataSampleDTO;
import cz.muni.fi.imageNet.downloader.enums.DownloadState;
import cz.muni.fi.imageNet.downloader.manager.ImageDownloadManager;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Class contains methods for processing of DataSampleDTO for other purpose.
 * 
 * @author Jakub Peschel
 */
public class DataSampleProcessor {
    final Configuration config;

    public DataSampleProcessor(Configuration config) {
        this.config = config;
    }
    
    public List<DataSample> getDataSampleCollection(DataSampleDTO[] dataSamples) {
        List<DataSample> result = new ArrayList();
        List<UrlImage> urlImages = new ArrayList();
        ImageDownloadManager downloadManager = new ImageDownloadManager(config);
        for (DataSampleDTO dataSample : dataSamples) {
            urlImages.add(transformToUrlImage(dataSample));
        }
        result.addAll(transformToDataSamples(downloadManager.processRequest(urlImages)));
        return result;
    }

    public List<Label> getDataSampleLabels(DataSampleDTO[] dataSamples) {
        Set<Label> labels = new TreeSet<Label>();
        for (DataSampleDTO dataSample : dataSamples) {
            for (String labelName : dataSample.getLabels()) {
                labels.add(new Label(labelName));
            }
        }
        return new ArrayList(labels);
    }

    public List<? extends DataSample> transformToDataSamples(DownloadResult processRequest) {
        List<DataSample> result = new ArrayList<DataSample>();
        for (DataImage dataImage : processRequest.getDataImageList()) {
            //TODO: vyřešit problémy s nestaženými
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

    public UrlImage transformToUrlImage(DataSampleDTO dataSample) {
        Set<Label> labels = new TreeSet();
        for (String labelName : dataSample.getLabels()) {
            labels.add(new Label(labelName));
        }
        return new UrlImage(dataSample.getUrl(), labels);
    }
}
