package cz.muni.fi.imagenet.image.net.cli;

import com.sampullara.cli.Argument;
import cz.muni.fi.image.net.core.enums.ModelType;

/**
 * CLI-Parser class for parsing arguments
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class ArgLoader {

    @Argument(alias = "dataset", description = "Location of csv", required = false)
    public static String datasetLoc;
    @Argument(alias = "imageLoc", description = "Location of downloaded images", required = false)
    public static String imageLoc;
    @Argument(alias = "type", description = "Type of modelWrapper for transfer learning", required = false)
    public static String model;
    @Argument(alias = "name", description = "Name of stored modelWrapper", required = false)
    public static String modelName;
    @Argument(alias = "time", description = "Time for computation in minutes", required = false)
    public static Long time;
    @Argument(alias = "logLocation", description = "Loaction of additional logfile", required = false)
    public static String logLocation;
    @Argument(alias = "method", description = "Run specific test method", required = false)
    public static String method;
    @Argument(alias = "imageURI", description = "Location of image for classification", required = false)
    public static String imageURI;
    @Argument(alias = "labelList", description = "List of labels", required = false, delimiter = ",")
    public static String[] labelList;
    @Argument(alias = "modelLoc", description = "Trained modelWrapper location", required = false)
    public static String modelLoc;
    @Argument(alias = "learningRate", description = "learningRate", required = false)
    public static Double learningRate;
    @Argument(alias = "epochCount", description = "amount of epochs", required = false)
    public static Integer epochCount;
    @Argument(alias = "tempLoc", description = "Temp location", required = false)
    public static String tempLoc;
    @Argument(alias = "gpuCount", description = "amount of gpu's", required = false)
    public static Integer gpuCount;
    @Argument(alias = "batchSize", description = "batch size", required = false)
    public static Integer batchSize;
    @Argument(alias = "l1", description = "l1", required = false)
    public static Double l1;
    @Argument(alias = "l2", description = "l2", required = false)
    public static Double l2;
    @Argument(alias = "ol2", description = "output l2", required = false)
    public static Double ol2;
    @Argument(alias = "dropout", description = "dropout", required = false)
    public static Double dropout;
    @Argument(alias = "oLearningRate", description = "olearningRate", required = false)
    public static Double oLearningRate;
    @Argument(alias = "jv", description = "Java minor version", required = false)
    public static Double javaVersion;
    @Argument(alias = "Dorg.bytedeco.javacpp.maxbytes", description = "ignored", required = false)
    public static String maxbytes;
    @Argument(alias = "Dorg.bytedeco.javacpp.maxPhysicalBytes", description = "ignored", required = false)
    public static String maxPhysicalBytes;
}
