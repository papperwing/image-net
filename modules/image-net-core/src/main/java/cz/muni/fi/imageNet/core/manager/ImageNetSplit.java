package cz.muni.fi.imageNet.core.manager;

import cz.muni.fi.imageNet.core.objects.DataSample;
import cz.muni.fi.imageNet.core.objects.DataSet;
import cz.muni.fi.imageNet.core.objects.Label;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.datavec.api.split.BaseInputSplit;
import org.datavec.api.split.InputSplit;

/**
 * Class is used for alternation of loading data into record reader.
 *
 * @author Jakub Peschel
 */
class ImageNetSplit
        extends BaseInputSplit
        implements InputSplit {

    private final DataSet dataSet;
    // Use for Collections, pass in list of file type strings
    protected String[] allowFormat = null;
    protected boolean recursive = true;
    protected Random random;
    protected boolean randomize = false;

    protected Map<URI, Set<Label>> labelMap = new HashMap<URI, Set<Label>>();

    protected ImageNetSplit(DataSet dataSet, String[] allowFormat, boolean recursive, Random random, boolean runMain) {
        this.allowFormat = allowFormat;
        this.dataSet = dataSet;
        this.recursive = recursive;
        if (random != null) {
            this.random = random;
            this.randomize = true;
        }
        if (runMain) {
            this.initialize();
        }
    }

    public ImageNetSplit(DataSet dataSet) {
        this(dataSet, null, true, null, true);
    }

    public ImageNetSplit(DataSet dataSet, Random rng) {
        this(dataSet, null, true, rng, true);
    }

    public ImageNetSplit(DataSet dataSet, String[] allowFormat) {
        this(dataSet, allowFormat, true, null, true);
    }

    public ImageNetSplit(DataSet dataSet, String[] allowFormat, Random rng) {
        this(dataSet, allowFormat, true, rng, true);
    }

    public ImageNetSplit(DataSet dataSet, String[] allowFormat, boolean recursive) {
        this(dataSet, allowFormat, recursive, null, true);
    }

    protected void initialize() {
        if (dataSet == null) {
            throw new IllegalArgumentException("Dataset must not be null.");
        }
        if (dataSet.getData().isEmpty()) {
            throw new IllegalArgumentException("Dataset must contain data.");
        }
        List<DataSample> samples = new ArrayList(dataSet.getData());
        if (randomize) {
            Collections.shuffle(samples, random);
        }

        uriStrings = new ArrayList<String>();
        labelMap = new HashMap<URI, Set<Label>>();
        
        for (final DataSample sample : samples) {
            final URI uri = new File(sample.getImageLocation()).toURI();
            uriStrings.add(uri.toString());
            labelMap.put(
                    uri,
                    sample.getLabelSet()
            );
        }

        this.length = samples.size();
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    public Map<URI, Set<Label>> getLabelMap() {
        return labelMap;
    }

    public void reset() {
        this.initialize();
    }
    
    public DataSet getDataSet(){
        return this.dataSet;
    }
}
