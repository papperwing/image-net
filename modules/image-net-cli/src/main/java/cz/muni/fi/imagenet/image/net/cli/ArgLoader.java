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
    @Argument(alias = "time", description="Time for computation (format: HH:MM:SS)", required = false)
    public static Long time;
    @Argument(alias = "logLocation", description="Loaction of additional logfile", required = false)
    public static String logLocation;
    @Argument(alias = "testMethod", description="Run specific test method", required = false)
    public static String testMethod;
}
