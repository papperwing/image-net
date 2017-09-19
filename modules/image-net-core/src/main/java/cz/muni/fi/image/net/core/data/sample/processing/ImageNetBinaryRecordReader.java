package cz.muni.fi.image.net.core.data.sample.processing;

import cz.muni.fi.image.net.core.data.normalization.ImageNormalizer;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.image.transform.ImageTransformator;
import cz.muni.fi.image.net.core.manager.LabelHelper;
import cz.muni.fi.image.net.core.objects.BinaryDataSample;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.util.*;

public class ImageNetBinaryRecordReader extends BaseRecordReader {

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
    DataSet dataSet;
    Iterator<DataSample> iterator;

    DataNormalization dataNormalizer;
    BinaryDataSample currentSample;

    ImageTransformator transformator;
    ImageNormalizer normalizer;

    public ImageNetBinaryRecordReader() {

    }

    public ImageNetBinaryRecordReader(int imageWidth, int imageHeight, int imageChannel, ModelType modelType) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageChannel = imageChannel;
        this.transformator =  new ImageTransformator(modelType, new int[] {imageWidth, imageHeight, imageChannel});
        this.normalizer = new ImageNormalizer(modelType);
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
        List<DataSample> randomList = new ArrayList(dataSet.getData());
        Collections.shuffle(randomList);
        this.iterator = randomList.iterator();
        if (dataNormalizer != null) {
            this.dataNormalizer = dataNormalizer;
        } else {
            this.dataNormalizer = normalizer.getDataNormalization();
        }
    }

    @Override
    public List<Writable> next() {
        if (iterator != null && iterator.hasNext()) {

            List<Writable> ret;

            currentSample = iterator.next();
            try {
                ret = transformator.transformImage(currentSample.getImageLocation());


                if(currentSample.)
                    ret.add(intWritable);
                }
                return ret;
            } catch (IOException ex) {
                logger.error("Loading of image " + currentSample.getImageLocation() + " was not sucessfull.", ex);
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
        return Arrays.asList(new String[] {"outside"});
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

    public List<Writable> record(ImageNetRecordMetaData metaData) {
        DataSample sample = dataSet.getData().get(Integer.parseInt(metaData.getLocation()));
        try {

            List<Writable> ret = transformator.transformImage(currentSample.getImageLocation());
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
            logger.error("Loading of image " + currentSample.getImageLocation() + " was not sucessfull.", ex);
            throw new IllegalStateException("Pointer Ex", ex);
        }
    }

    @Override
    public Record nextRecord() {
        List<Writable> list = next();
        return new org.datavec.api.records.impl.Record(list, new ImageNetRecordMetaData(dataSet.getData().indexOf(currentSample)));
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> out = new ArrayList<>();
        for (RecordMetaData meta : recordMetaDatas) {
            List<Writable> next;
            if (meta instanceof ImageNetRecordMetaData) {
                next = record((ImageNetRecordMetaData) meta);
            } else {
                URI uri = meta.getURI();
                File f = new File(uri);

                try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
                    next = record(uri, dis);
                }
            }
            out.add(new org.datavec.api.records.impl.Record(next, meta));
        }
        return out;
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
