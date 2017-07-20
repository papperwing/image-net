package cz.muni.fi.image.net.core.image;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.*;
import java.util.Arrays;

/**
 * FrameConverter for {@link INDArray}
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class INDArrayFrameConverter extends FrameConverter<INDArray> {
    @Override
    public Frame convert(INDArray indarray) {
        if (indarray == null) {
            return null;
        }

        //if (indarray.rank() > 3 || indarray.rank() < 2)
        //    throw new IllegalArgumentException("INDArray doesnt have right amount of dimensions: " + indarray.rank());
        int[] shape = indarray.shape();

        int width = 0;
        int height = 0;
        int depth = Frame.DEPTH_BYTE;
        int channel = 1;
        switch (shape.length) {
            case 4:
            case 3:
                channel = shape[1];
                if (channel < 3 || channel > 4) throw new IllegalArgumentException("Channels must be in RGB or RGBA");
            case 2:
                width = shape[2];
                height = shape[3];
                break;
            default:
                throw new IllegalArgumentException("Rank and Shape doesnt match one other");
        }


        frame = new Frame(width, height, depth, channel);
        ByteBuffer out = ByteBuffer.allocate(width * height * channel);
        System.out.println(Arrays.asList(out.array()).toString());
        restoreRGBImage(indarray, out);
        frame.image[0] = out;
        return frame;
    }

    @Override
    public INDArray convert(Frame frame) {
        throw new UnsupportedOperationException("Not implemented yet because I didnt needed it");
    }


    /**
     * Returns RGB image out of 3D tensor
     *
     * @param tensor3D
     * @return
     */
    private void restoreRGBImage(INDArray tensor3D, ByteBuffer out) {

        normalize(tensor3D);

        INDArray arrayR = null;
        INDArray arrayG = null;
        INDArray arrayB = null;

        // entry for 3D input vis
        if (tensor3D.shape()[1] == 3) {
            arrayR = tensor3D.tensorAlongDimension(2, 2, 3);
            arrayG = tensor3D.tensorAlongDimension(1, 2, 3);
            arrayB = tensor3D.tensorAlongDimension(0, 2, 3);
        } else {
            // for all other cases input is just black & white, so we just assign the same channel data to RGB, and represent everything as RGB
            arrayB = tensor3D.tensorAlongDimension(0, 2, 3);
            arrayG = arrayB;
            arrayR = arrayB;
        }

        for (int y = 0; y < arrayR.rows(); y++) {
            for (int x = 0; x < arrayR.columns(); x++) {
                byte valueX = (byte) (arrayR.getRow(y).getInt(x));
                byte valueY = (byte) (arrayG.getRow(y).getInt(x));
                byte valueZ = (byte) (arrayB.getRow(y).getInt(x));

                out.put(new byte[]{valueZ, valueY, valueX});
            }
        }
    }

    public void normalize(INDArray tensor3D) {
        int max = tensor3D.amaxNumber().intValue();
        int min = tensor3D.aminNumber().intValue();
        if (max > 255 | min < 0) {
            tensor3D.sub(min);
            tensor3D.mul(255 / (max - min));
        }
        tensor3D.sub(256/2);
    }

}
