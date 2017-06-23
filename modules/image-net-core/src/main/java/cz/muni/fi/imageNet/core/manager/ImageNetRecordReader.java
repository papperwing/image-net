package cz.muni.fi.imageNet.core.manager;

import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.Label;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Jakub Peschel
 */
public class ImageNetRecordReader extends BaseRecordReader {

    private final Logger logger = LoggerFactory.getLogger(ImageNetRecordReader.class);

    /**
     * Image will be resized to this width
     */
    int imageWidth;
    /**
     * Image will be resized to this height
     */
    int imageHeight;
    /**
     * Image will use this amount of channels
     */
    int imageChannel;

    Configuration conf;
    List<Label> labels;
    DataSet dataSet;
    Iterator<DataSample> iterator;

    BaseImageLoader imageLoader;

    DataNormalization dataNormalizer;

    public ImageNetRecordReader() {

    }

    public ImageNetRecordReader(int imageWidth, int imageHeight, int imageChannel, ImageTransform transform) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageChannel = imageChannel;
        this.imageLoader = new NativeImageLoader(imageHeight, imageWidth, imageChannel, transform);
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        throw new UnsupportedOperationException("ImageNetRecordReader doesnt use input split");
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.conf = conf;
        this.initialize(split);
    }

    public void initialize(Configuration conf, DataSet dataSet, DataNormalization dataNormalizer) {
        this.conf = conf;
        this.initialize(dataSet, dataNormalizer);
    }

    public void initialize(DataSet dataSet, DataNormalization dataNormalizer) {
        this.dataSet = dataSet;
        this.labels = dataSet.getLabels();
        List<DataSample> randomList = new ArrayList(dataSet.getData());
        Collections.shuffle(randomList);
        this.iterator = randomList.iterator();
        if (dataNormalizer != null) {
            this.dataNormalizer = dataNormalizer;
        } else {
            this.dataNormalizer = new ImagePreProcessingScaler(0, 1);
        }
    }

    @Override
    public List<Writable> next() {
        if (iterator != null && iterator.hasNext()) {

            List<Writable> ret;

            DataSample sample = iterator.next();
            File imageFile = new File(sample.getImageLocation());
            try {
                invokeListeners(imageFile);
                INDArray row = imageLoader.asMatrix(imageFile);

                dataNormalizer.transform(row);

                ret = RecordConverter.toRecord(row);
                for (Label label : labels) {
                    //TODO: this part is specifical for binary multi-label usage. Need to be rewriten for general imageclassification usage
                    final IntWritable intWritable;
                    if (sample.getLabelSet().contains(label)) {
                        intWritable = new IntWritable(1);
                    } else {
                        intWritable = new IntWritable(0);
                    }
                    ret.add(intWritable);
                }
                return ret;
            } catch (IOException ex) {
                logger.error("Loading of image " + imageFile.toString() + " was not sucessfull.", ex);
            }
        }
        throw new IllegalStateException("No more elements");
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public List<String> getLabels() {
        return LabelHelper.translate(labels);
    }

    @Override
    public void reset() {
        List<DataSample> randomList = new ArrayList(dataSet.getData());
        Collections.shuffle(randomList);
        this.iterator = randomList.iterator();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Record nextRecord() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * This RecordReader is not working with streams. Method do nothing.
     *
     * {@link Closeable#close()}
     */
    @Override
    public void close() throws IOException {
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

}
