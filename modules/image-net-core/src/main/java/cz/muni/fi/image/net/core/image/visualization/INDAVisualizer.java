package cz.muni.fi.image.net.core.image.visualization;

import cz.muni.fi.image.net.core.image.INDArrayFrameConverter;
import org.bytedeco.javacv.CanvasFrame;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class INDAVisualizer {

    public void visualizeINDA(final INDArray indArray) {
        this.visualizeINDA(indArray, null);
    }

    public void visualizeINDA(final INDArray indArray, String windowName) {
        final INDArrayFrameConverter indaConverter = new INDArrayFrameConverter();
        if (windowName == null) {
            windowName = "";
        }

        final int width = indArray.shape()[2];
        final int height = indArray.shape()[3];

        final CanvasFrame canvas = new CanvasFrame(windowName, 1.0);
        canvas.setCanvasSize(width, height);
        canvas.showImage(indaConverter.convert(indArray));
    }

}
