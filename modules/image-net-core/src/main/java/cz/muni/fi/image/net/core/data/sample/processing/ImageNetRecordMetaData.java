package cz.muni.fi.image.net.core.data.sample.processing;

import cz.muni.fi.image.net.core.objects.DataSet;

import java.net.URI;

import org.datavec.api.records.metadata.RecordMetaData;

/**
 * Metadata containing location in {@link DataSet}
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageNetRecordMetaData implements RecordMetaData {


    int dataSetIndex;

    /**
     * Constructor of {@link ImageNetRecordMetaData}
     *
     * @param dataSetIndex index pointing inside {@link DataSet}
     */
    public ImageNetRecordMetaData(int dataSetIndex) {
        this.dataSetIndex = dataSetIndex;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getLocation() {
        return "" + dataSetIndex;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public URI getURI() {
        throw new UnsupportedOperationException("URI doesnt exist because actual dataset is in memory");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<?> getReaderClass() {
        return this.getClass();
    }

}
