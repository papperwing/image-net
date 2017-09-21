package cz.muni.fi.image.net.core.data.sample.processing;

import cz.muni.fi.image.net.core.data.normalization.ImageNormalizer;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.image.transform.ImageTransformator;
import cz.muni.fi.image.net.core.manager.LabelHelper;
import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;

import java.io.BufferedInputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
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
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Custom implementation of {@link org.datavec.api.records.reader.RecordReader} for this library.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
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

    DataNormalization dataNormalizer;
    DataSample currentSample;

    ImageTransformator transformator;
    ImageNormalizer normalizer;

    public ImageNetRecordReader() {

    }

    public ImageNetRecordReader(
            final int imageWidth,
            final int imageHeight,
            final int imageChannel,
            final ModelType modelType
    ) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageChannel = imageChannel;
        this.transformator = new ImageTransformator(modelType, new int[]{imageWidth, imageHeight, imageChannel});
        this.normalizer = new ImageNormalizer(modelType);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(final InputSplit split) throws IOException, InterruptedException {
        throw new UnsupportedOperationException("ImageNetRecordReader doesnt use input split");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(
            final Configuration conf,
            final InputSplit split
    ) throws IOException, InterruptedException {
        this.conf = conf;
        this.initialize(split);
    }

    /**
     * Initialization of {@link ImageNetRecordReader}
     *
     * @param conf           {@link Configuration}
     * @param dataSet        {@link DataSet}
     * @param dataNormalizer {@link DataNormalization}
     */
    public void initialize(
            final Configuration conf,
            final DataSet dataSet,
            final DataNormalization dataNormalizer
    ) {
        this.conf = conf;
        this.initialize(dataSet, dataNormalizer);
    }

    /**
     * Initialization of {@link ImageNetRecordReader}
     *
     * @param dataSet        {@link DataSet}
     * @param dataNormalizer {@link DataNormalization}
     */
    public void initialize(
            final DataSet dataSet,
            final DataNormalization dataNormalizer
    ) {
        this.dataSet = dataSet;
        this.labels = dataSet.getLabels();
        List<DataSample> randomList = new ArrayList(dataSet.getData());
        Collections.shuffle(randomList);
        this.iterator = randomList.iterator();
        if (dataNormalizer != null) {
            this.dataNormalizer = dataNormalizer;
        } else {
            this.dataNormalizer = normalizer.getDataNormalization();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Writable> next() {
        if (iterator != null && iterator.hasNext()) {

            List<Writable> ret;

            currentSample = iterator.next();
            try {
                ret = transformator.transformImage(currentSample.getImageLocation());

                for (Label label : labels) {
                    //TODO: this part is specifical for binary multi-label usage. Need to be rewriten for general imageclassification usage
                    final IntWritable intWritable;
                    if (currentSample.getLabelSet().contains(label)) {
                        intWritable = new IntWritable(1);
                    } else {
                        intWritable = new IntWritable(0);
                    }
                    ret.add(intWritable);
                }
                return ret;
            } catch (IOException ex) {
                logger.error("Loading of image " + currentSample.getImageLocation() + " was not sucessfull.", ex);
            }
        }
        throw new IllegalStateException("No more elements");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<String> getLabels() {
        return LabelHelper.translate(labels);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void reset() {
        List<DataSample> randomList = new ArrayList(dataSet.getData());
        Collections.shuffle(randomList);
        this.iterator = randomList.iterator();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Writable> record(
            final URI uri,
            final DataInputStream dataInputStream
    ) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public List<Writable> record(final ImageNetRecordMetaData metaData) {
        final DataSample sample = dataSet.getData().get(Integer.parseInt(metaData.getLocation()));
        try {

            final List<Writable> ret = transformator.transformImage(currentSample.getImageLocation());
            for (final Label label : labels) {
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
    public Record loadFromMetaData(final RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(final List<RecordMetaData> recordMetaDatas) throws IOException {
        final List<Record> out = new ArrayList<>();
        for (final RecordMetaData meta : recordMetaDatas) {
            List<Writable> next;
            if (meta instanceof ImageNetRecordMetaData) {
                next = record((ImageNetRecordMetaData) meta);
            } else {
                final URI uri = meta.getURI();
                final File f = new File(uri);

                try (final DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
                    next = record(uri, dis);
                }
            }
            out.add(new org.datavec.api.records.impl.Record(next, meta));
        }
        return out;
    }

    /**
     * This RecordReader is not working with streams. Method do nothing.
     * <p>
     * {@link Closeable#close()}
     */
    @Override
    public void close() throws IOException {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Configuration getConf() {
        return conf;
    }

}
