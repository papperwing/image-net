package cz.muni.fi.image.net.core.dataset.processor;

import cz.muni.fi.image.net.core.data.normalization.ImageNormalizer;
import cz.muni.fi.image.net.core.data.sample.processing.ImageNetRecordReader;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.objects.DataSet;
import cz.muni.fi.image.net.core.enums.ModelType;
import org.datavec.api.berkeley.Pair;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jpeschel on 4.7.17.
 */
public class DataSetProcessor {

    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    final Configuration conf;

    //TODO: need to be changed based on input shape of specific network
    private final int height = 224;
    private final int width = 224;
    private final int channels = 3;

    final ModelType modelType;

    public DataSetProcessor(Configuration conf,
                            ModelType modelType) {
        this.conf = conf;
        this.modelType = modelType;
    }

    public DataSetIterator prepareDataSetIterator(
            final DataSet dataset
    ) {
        final List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();

        pipeline.add(
                new Pair(
                        new ResizeImageTransform(
                                300,
                                300
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new RandomCropTransform(
                                height,
                                width
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new FlipImageTransform(
                                1
                        ),
                        0.5
                )
        );

        final ImageTransform combinedTransform = new MultiImageTransform(
                new ImageTransform[]{
                        new PipelineImageTransform(
                                pipeline,
                                false
                        )
                }
        );

        final ImageNetRecordReader recordReader = new ImageNetRecordReader(
                height,
                width,
                channels,
                combinedTransform,
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

        DataNormalization normalization = new ImageNormalizer(modelType).getDataNormalization();
        normalization.fit(dataIter);
        dataIter.setPreProcessor(normalization);

        return dataIter;
    }

    public DataSetIterator presaveDataSetIterator(
            DataSetIterator dataIter,
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
            next.save(
                    new File(
                            saveFolder,
                            saveDataName + "-" + dataSaved + ".bin"
                    )
            );
            dataSaved++;
        }
        logger.debug("DataSet presaved");

        DataSetIterator iter = new ExistingMiniBatchDataSetIterator(
                saveFolder,
                saveDataName + "-%d.bin"
        );

        DataNormalization normalization = new ImageNormalizer(modelType).getDataNormalization();
        normalization.fit(iter);
        iter.setPreProcessor(normalization);
        return iter;
    }
}
