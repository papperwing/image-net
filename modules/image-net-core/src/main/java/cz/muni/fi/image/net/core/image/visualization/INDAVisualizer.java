package cz.muni.fi.image.net.core.image.visualization;

import cz.muni.fi.image.net.core.image.INDArrayFrameConverter;
import org.bytedeco.javacv.CanvasFrame;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Jakub Peschel
 */
public class INDAVisualizer {

    public void visualizeINDA(INDArray indArray){
        this.visualizeINDA(indArray, null);
    }
    public void visualizeINDA(INDArray indArray, String windowName){
        INDArrayFrameConverter indaConverter = new INDArrayFrameConverter();
        if(windowName == null){
            windowName = "";
        }

        int width = indArray.shape()[2];
        int height = indArray.shape()[3];

        CanvasFrame canvas = new CanvasFrame(windowName, 1.0);
        canvas.setCanvasSize(width,height);
        canvas.showImage(indaConverter.convert(indArray));
    }

}
