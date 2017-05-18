package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.Label;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Jakub Peschel
 */
public class ImageNetRecordReader
        extends BaseRecordReader {

    private final Logger logger = LoggerFactory.getLogger(ImageNetRecordReader.class);

    protected int height;
    protected int width;
    protected int channels = 1;
    protected ImageTransform imageTransform;
    protected Iterator<File> iter;
    protected File currentFile;
    protected BaseImageLoader imageLoader;
    protected Map<URI, Set<Label>> labelMap;
    protected Set<Label> labels;
    protected InputSplit inputSplit;
    protected Configuration conf;
    protected List<File> allFiles;

    public ImageNetRecordReader(
            int height,
            int width,
            int channels
    ) {
        this(height, width, channels, null);
    }

    public ImageNetRecordReader(
            int height,
            int width,
            int channels,
            ImageTransform imageTransform) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.imageTransform = imageTransform;

        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
    }

    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        throw new UnsupportedOperationException("Configuration is set in different way");
    }

    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (!(split instanceof ImageNetSplit)) {
            throw new UnsupportedOperationException("Not available to use ImageNetRecordReader with simple split.");
        }
        this.inputSplit = split;
        ImageNetSplit isplit = (ImageNetSplit) split;
        this.labelMap = isplit.getLabelMap();
        
        URI[] locations = split.locations();
        this.allFiles = new ArrayList<>();
        for (URI location : locations) {
            File imgFile = new File(location);
            allFiles.add(imgFile);
        }
        this.labels = new TreeSet(isplit.getDataSet().getLabels());

        iter = allFiles.iterator();

    }

    public List<Writable> next() {
        if (iter != null && iter.hasNext()) {
        
            List<Writable> ret = new ArrayList();
            File imageFile = iter.next();;
            currentFile = imageFile;
            try {
                invokeListeners(imageFile);
                INDArray row = imageLoader.asMatrix(imageFile);
                ret = RecordConverter.toRecord(row);
                System.gc();//asMatrix probably cause increase of memory usage
                for (Label label : labelMap.get(imageFile.toURI())) {
                    final int indexOf = getLabels().indexOf(label.getLabelName());
                    final IntWritable intWritable = new IntWritable(indexOf);
                    ret.add(intWritable);
                }
                return ret;
            } catch (IOException ex) {
                logger.error("Loading of image " + imageFile.toString() + " was not sucessfull.", ex);
            }
        }
        throw new IllegalStateException("No more elements");

    }

    public boolean hasNext() {
        return iter.hasNext();
    }

    /**
     * Method return names of custom Label objects.
     *
     * @return list of unique names
     */
    public List<String> getLabels() {
        List<String> result = new ArrayList<String>();
        for (Label label : labels) {
            result.add(label.getLabelName());
        }
        return result;
    }

    public void reset() {
        if (inputSplit == null) {
            throw new IllegalStateException("Cannot reset without first initializing");
        }
        try {
            initialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during ImageNetRecordReader reset", e);
        }
    }

    public void close() throws IOException {

    }

    public void setConf(Configuration conf) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public Configuration getConf() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public Record nextRecord() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
