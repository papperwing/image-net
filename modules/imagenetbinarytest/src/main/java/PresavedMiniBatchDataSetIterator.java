package cz.muni.fi.image.net.core.data.sample.processing;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Custom implementation of {@link DataSetIterator} for this library.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class PresavedMiniBatchDataSetIterator implements DataSetIterator {

    public static Logger logger = LoggerFactory.getLogger(PresavedMiniBatchDataSetIterator.class);

    public static final String DEFAULT_PATTERN = "dataset-[0-9]+.bin";

    private final File rootDir;
    private final List<File> miniBatches;
    private Iterator<File> iterator;
    private File currentFile = null;
    private DataSetPreProcessor dataSetPreProcessor;
    private final String pattern;
    private final boolean shuffle;

    /**
     * Create with the given root directory, using the default filename pattern {@link #DEFAULT_PATTERN}
     *
     * @param rootDir the root directory to use
     */
    public PresavedMiniBatchDataSetIterator(
            final File rootDir
    ) {
        this(rootDir, DEFAULT_PATTERN);
    }

    public PresavedMiniBatchDataSetIterator(
            final File rootDir,
            final String pattern
    ) {
        this(rootDir, pattern, false);
    }

    /**
     * @param rootDir The root directory to use
     * @param pattern The filename pattern to use. Used with {@code String.format(pattern,idx)}, where idx is an
     *                integer, starting at 0.
     * @param
     */
    public PresavedMiniBatchDataSetIterator(
            final File rootDir,
            final String pattern,
            final boolean suffle
    ) {

        this.rootDir = rootDir;
        this.pattern = pattern;
        this.shuffle = suffle;
        miniBatches = new ArrayList<>();

        final FilenameFilter fileFilter = new FilenameFilter() {
            @Override
            public boolean accept(File file, String s) {
                return s.matches(pattern);
            }
        };

        final String[] filenames = rootDir.list(fileFilter);
        for (final String fileName : filenames) {
            miniBatches.add(new File(rootDir, fileName));
        }

        reset();
    }

    /**
     * <b>Not supported</b>
     * {@inheritDoc}
     */
    @Override
    public DataSet next(final int num) {
        throw new UnsupportedOperationException("Unable to load custom number of examples");
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if (shuffle) {
            Collections.shuffle(miniBatches);
        }
        iterator = miniBatches.iterator();
        this.currentFile = null;
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int cursor() {
        return this.currentFile == null ? 0 : miniBatches.indexOf(this.currentFile);
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(final DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public void remove() {
        //no opt;
    }

    @Override
    public DataSet next() {
        this.currentFile = iterator.next();
        final DataSet ret = new DataSet();
        ret.load(currentFile);
        if (dataSetPreProcessor != null)
            dataSetPreProcessor.preProcess(ret);

        return ret;
    }
}
