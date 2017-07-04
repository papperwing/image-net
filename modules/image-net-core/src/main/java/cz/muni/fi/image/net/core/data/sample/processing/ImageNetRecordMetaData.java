package cz.muni.fi.image.net.core.data.sample.processing;

import cz.muni.fi.image.net.core.objects.DataSet;

import java.net.URI;
import org.datavec.api.records.metadata.RecordMetaData;

/**
 * Metadata cointaing location in {@link DataSet}
 * 
 * @author Jakub Peschel
 */
public class ImageNetRecordMetaData implements RecordMetaData{

    
    int dataSetIndex;

    public ImageNetRecordMetaData(int dataSetIndex) {
        this.dataSetIndex = dataSetIndex;
    }
    
    @Override
    public String getLocation() {
        return "" + dataSetIndex;
    }

    @Override
    public URI getURI() {
        throw new UnsupportedOperationException("URI doesnt exist because actual dataset is in memory");
    }

    @Override
    public Class<?> getReaderClass() {
        return this.getClass();
    }

}
