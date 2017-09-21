package cz.muni.fi.image.net.dataset.dataset;

import cz.muni.fi.image.net.core.objects.DataSample;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.objects.Label;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of {@link DataSet}
 *
 * @author Jakub Peschel
 */
public class DataSetImpl implements DataSet {

    private List<DataSample> dataList;

    private final int dataSetLength;

    private final List<Label> labels;

    /**
     * Constructor  of {@link DataSetImpl}
     *
     * @param dataList {@link List} of {@link DataSample}s
     * @param labels   {@link List} of {@link Label}s
     */
    public DataSetImpl(
            final List<DataSample> dataList,
            final List<Label> labels
    ) {
        this.dataList = dataList;
        this.dataSetLength = dataList.size();
        this.labels = labels;
        for (final DataSample sample : dataList) {
            if (!labels.containsAll(sample.getLabelSet())) {
                throw new IllegalArgumentException("DataSample contains unknown label" + sample.toString());
            }
        }
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public List<DataSample> getData() {
        return Collections.unmodifiableList(dataList);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int length() {
        return this.dataSetLength;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Label> getLabels() {
        return Collections.unmodifiableList(labels);
    }

    /**
     * Getter of {@link Label#labelName}s
     *
     * @return {@link Label#labelName}s
     */
    public List<String> getLabelStrings() {
        List<String> result = new ArrayList();
        for (Label label : labels) {
            result.add(label.getLabelName());
        }
        return Collections.unmodifiableList(result);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DataSet split(double splitPercentage) {
        validatePecentage(splitPercentage);
        final int splitIndex = (int) Math.ceil(this.dataSetLength * splitPercentage);

        final List<DataSample> splitList = this.dataList.subList(splitIndex, dataSetLength - 1);
        dataList = this.dataList.subList(0, splitIndex);
        final DataSetImpl splitSet = new DataSetImpl(splitList, labels);

        return splitSet;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<Label, Integer> getLabelDistribution() {
        final Map<Label, Integer> result = new HashMap();
        for (final Label label : getLabels()) {
            result.put(label, new Integer(0));
        }

        for (final DataSample sample : getData()) {
            for (final Label label : sample.getLabelSet()) {
                result.put(label, result.get(label) + 1);
            }
        }

        return result;
    }

    private void validatePecentage(
            final double splitPercentage
    ) throws IllegalArgumentException {
        if (splitPercentage < 0 || splitPercentage > 1) {
            throw new IllegalArgumentException("Percentage must be between 0 and 1.");
        }
    }


}
