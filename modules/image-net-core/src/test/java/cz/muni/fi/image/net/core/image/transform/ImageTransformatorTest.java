package cz.muni.fi.image.net.core.image.transform;

import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.core.image.visualization.INDAVisualizer;
import cz.muni.fi.image.net.core.objects.Configuration;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.jetbrains.annotations.NotNull;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

/**
 * Created by jpeschel on 17.7.17.
 */
public class ImageTransformatorTest {

    @Ignore
    @Test
    public void transformVizualTest() throws Exception {
        System.out.println("Vizual test of transformation");
        final File image = new File(this.getClass().getClassLoader().getResource("testImage.jpg").getFile());

        Runnable originalThread = new Runnable() {
            @Override
            public void run() {
                try {
                    ImageTransform vizualizer = new ShowImageTransform("vizualizer-before", 100000);
                    new NativeImageLoader(224, 224, 3, vizualizer).asMatrix(image);
                } catch (Exception ex) {
                    System.err.println(
                            "Exception in original thread"
                    );
                }
            }
        };

        Runnable transformedThread = new Runnable() {
            @Override
            public void run() {
                try {
                    ImageTransform vizualizer = new ShowImageTransform("vizualizer-after", 100000);
                    ImageTransform transform = new PipelineImageTransform(new ImageTransformator(ModelType.RESNET50).getTransformation(ModelType.RESNET50), vizualizer);
                    new NativeImageLoader(224, 224, 3, transform).asMatrix(image);
                } catch (Exception ex) {
                    System.err.println(
                            "Exception in transformed thread"
                    );
                }
            }
        };

        ThreadPoolExecutor exec = new ThreadPoolExecutor(2, 2, 1000, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(2));
        exec.execute(originalThread);
        exec.execute(transformedThread);

        System.out.println("End of vizual test");
    }

    @Ignore
    @Test
    public void transform2VizualTest() throws Exception {
        System.out.println("Second vizual test of transformation");
        final File image = new File(this.getClass().getClassLoader().getResource("testImage.jpg").getFile());

        ImageTransform vizualizer = new ShowImageTransform("vizualizer-before");
        new NativeImageLoader(224, 224, 3, vizualizer).asMatrix(image);


        vizualizer = new ShowImageTransform("vizualizer-after", 10000);
        ImageTransform transform = new PipelineImageTransform(new ImageTransformator(ModelType.RESNET50).getTransformation(ModelType.RESNET50), vizualizer);
        new NativeImageLoader(224, 224, 3, transform).asMatrix(image);

        System.out.println("End of vizual test");
    }

    @Test
    public void visualizeINDATest() throws Exception {
        System.out.println("Third vizual test of transformation");
        final File imageFile = new File(this.getClass().getClassLoader().getResource("testImage.jpg").getFile());

        INDArray array = new NativeImageLoader(224, 224, 3).asMatrix(imageFile);

        INDAVisualizer visualizer = new INDAVisualizer();
        visualizer.visualizeINDA(array, "testINDA");


    }

}