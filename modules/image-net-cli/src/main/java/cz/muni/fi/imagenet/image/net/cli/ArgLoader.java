package cz.muni.fi.imagenet.image.net.cli;

import com.sampullara.cli.Argument;
import cz.muni.fi.imageNet.Pojo.ModelType;

/**
 * CLI-Parser class for parsing arguments
 * @author jpeschel
 */
public class ArgLoader {
    
    @Argument(alias = "dataset", description="Location of csv", required = true)
    public static String datasetLoc;
    @Argument(alias = "imageLoc", description="Location of downloaded images", required = false)
    public static String imageLoc;
    @Argument(alias = "type", description="Type of model for transfer learning", required = false)
    public static ModelType model;
    @Argument(alias = "name", description="Name of stored model", required = true)
    public static String modelName;
    @Argument(alias = "time", description="Time for computation in minutes", required = false)
    public static Long time;
    @Argument(alias = "logLocation", description="Loaction of additional logfile", required = false)
    public static String logLocation;
    @Argument(alias = "method", description="Run specific test method", required = true)
    public static String method;
    @Argument(alias = "imageURI", description="Location of image for classification", required = false)
    public static String imageURI;
    @Argument(alias = "labelList", description = "List of labels", required = false, delimiter = ",")
    public static String[] labelList;
    @Argument(alias = "modelLoc", description="Trained model location", required = false)
    public static String modelLoc;
    @Argument(alias = "learningRate", description="learningRate", required = false)
    public static Double learningRate;
}
