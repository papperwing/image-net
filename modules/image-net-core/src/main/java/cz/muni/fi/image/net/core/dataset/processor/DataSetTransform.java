package cz.muni.fi.image.net.core.dataset.processor;

import cz.muni.fi.image.net.core.data.normalization.ImageNormalizer;
import cz.muni.fi.image.net.core.data.sample.processing.ImageNetRecordReader;
import cz.muni.fi.image.net.core.data.sample.processing.PresavedMiniBatchDataSetIterator;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * DataSetTransform serves as transform from {@link DataSet} into {@link DataSetIterator}.
 * It also can presave data for sake of saving time and resources for
 * repeated transformation of images into {@link org.nd4j.linalg.api.ndarray.INDArray}.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class DataSetTransform {

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    private final Configuration conf;

    //TODO: need to be changed based on input shape of specific network
    private final int height = 224;
    private final int width = 224;
    private final int channels = 3;

    private final ModelType modelType;

    /**
     * Constructor of {@link DataSetTransform}
     *
     * @param conf      global configuration
     * @param modelType type of model for which is {@link DataSetIterator} created
     */
    public DataSetTransform(
            final Configuration conf,
            final ModelType modelType
    ) {
        this.conf = conf;
        this.modelType = modelType;
    }

    /**
     * Transformation of {@link DataSet} into {@link DataSetIterator}.
     *
     * @param dataset {@link DataSet} containing data for training
     * @return {@link DataSetIterator} containing data for training
     */
    public DataSetIterator prepareDataSetIterator(
            final DataSet dataset
    ) {

        final ImageNetRecordReader recordReader = new ImageNetRecordReader(
                height,
                width,
                channels,
                modelType
        );

        logger.debug("Record reader inicialization.");
        recordReader.initialize(
                dataset,
                null
        );

        final DataSetIterator dataIter = new AsyncDataSetIterator(
                new RecordReaderDataSetIterator(
                        recordReader,
                        this.conf.getBatchSize(),
                        1,
                        dataset.getLabels().size(),
                        true
                )
        );

        dataIter.setPreProcessor(new ImageNormalizer(modelType).getDataNormalization());

        return dataIter;
    }

    /**
     * Saving
     *
     * @param dataIter
     * @param modelType
     * @param saveDataName
     * @return
     */
    public DataSetIterator presaveDataSetIterator(
            final DataSetIterator dataIter,
            final ModelType modelType,
            final String saveDataName
    ) {
        logger.debug("PreSaving dataset for faster processing");
        final File saveFolder = new File(
                this.conf.getTempFolder()
                        + File.separator
                        + "minibatches"
                        + File.separator
                        + saveDataName
        );
        saveFolder.mkdirs();
        saveFolder.deleteOnExit();
        int dataSaved = 0;
        while (dataIter.hasNext()) {
            final org.nd4j.linalg.dataset.DataSet next = dataIter.next();

            logger.debug("" + dataSaved);
            final File saveFile = new File(
                    saveFolder,
                    saveDataName + "-" + dataSaved + ".bin"
            );
            saveFile.deleteOnExit();
            next.save(
                    saveFile
            );
            dataSaved++;
        }
        logger.debug("DataSet presaved");

        final DataSetIterator iter = new PresavedMiniBatchDataSetIterator(
                saveFolder,
                saveDataName + "-[0-9]+.bin",
                true
        );
        return iter;
    }
}
