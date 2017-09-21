package cz.muni.fi.image.net.core.image.transform;

import cz.muni.fi.image.net.core.enums.ModelType;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ImageTransformator {

    DataNormalization normalizer = null;

    BaseImageLoader loader;
    int[] inputShape;

    public ImageTransformator(final ModelType modelType) {
        this(
                modelType,
                new int[]{224, 224, 3}
        );
    }

    public ImageTransformator(
            final ModelType modelType,
            final int[] inputShape
    ) {
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Expected length of inputShape 3 but got: " + inputShape.length);
        }
        this.inputShape = inputShape;
        this.loader = new NativeImageLoader(
                inputShape[0],
                inputShape[1],
                inputShape[2],
                getTransformation(modelType)
        );
    }

    public List<Writable> transformImage(final String fileLocation) throws IOException {

        final File imageFile = new File(fileLocation);
        final INDArray row = loader.asMatrix(imageFile);

        if (normalizer != null) {
            normalizer.transform(row);

        }

        return RecordConverter.toRecord(row);
    }


    public ImageTransform getTransformation(final ModelType modelType) {
        final List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        int biggerWidth = (int) (inputShape[0] * 1.35);//multiplier just for sake of getting cca 300 pixel size before random cropping
        int biggerHeight = (int) (inputShape[1] * 1.35);//multiplier just for sake of getting cca 300 pixel size before random cropping
        //TODO: add sampling option for unbalanced classes
        pipeline.add(
                new Pair(
                        new ResizeImageTransform(
                                biggerWidth,
                                biggerHeight
                        ),
                        1.0
                )
        );

        pipeline.add(
                new Pair(
                        new RandomCropTransform(
                                inputShape[0],
                                inputShape[1]
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
                        /*,
                        new ShowImageTransform("Title", 10)*/
                }
        );
        return combinedTransform;
    }

}
