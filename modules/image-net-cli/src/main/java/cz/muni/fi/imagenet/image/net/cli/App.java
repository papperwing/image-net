package cz.muni.fi.imagenet.image.net.cli;

import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.core.FileAppender;
import ch.qos.logback.classic.LoggerContext;
import cz.muni.fi.imageNet.POJO.DataSampleDTO;
import cz.muni.fi.imageNet.Pojo.Configuration;
import cz.muni.fi.imageNet.Pojo.ModelType;
import cz.muni.fi.imageNet.Service.ImageNetAPI;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sampullara.cli.Args;

/**
 *
 * @author Jakub Peschel
 */
public class App {

    public static Logger logger = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws Exception {
        final List<String> parse;
        try {
            parse = Args.parse(ArgLoader.class, args);
        } catch (IllegalArgumentException e) {
            Args.usage(ArgLoader.class);
            System.exit(1);
            return;
        }
        logger.info("Started command line interface.");
        File datasetFile = new File(ArgLoader.datasetLoc);
        if (!datasetFile.isFile() || !datasetFile.canRead()) {
            throw new IllegalArgumentException("There is wrong path to dataset file.");
        }

        Configuration config = new Configuration();

        if (ArgLoader.imageLoc != null) {
            config.setImageDownloadFolder(ArgLoader.imageLoc);
        } else {
            config.setImageDownloadFolder("/home/jpeschel/static/");
        }

        if (ArgLoader.time != null) {
            config.setTimed(true);
            config.setTime(ArgLoader.time);
        }

        ImageNetAPI api = new ImageNetAPI(config);

        switch (ArgLoader.method) {
            case "TRAIN":
                logger.info("Loading dataset: " + datasetFile.getAbsolutePath());

                List<DataSampleDTO> datasetList = new ArrayList();

                try (final BufferedReader fileReader = new BufferedReader(new FileReader(datasetFile))) {
                    String line = fileReader.readLine();
                    while (line != null) {

                        DataSampleDTO sample = new DataSampleDTO(line);
                        datasetList.add(sample);
                        line = fileReader.readLine();
                    }
                }

                DataSampleDTO[] dataset = new DataSampleDTO[datasetList.size()];
                dataset = datasetList.toArray(dataset);

                api.getModel(
                        ArgLoader.modelName,
                        dataset,
                        7,
                        ArgLoader.model != null ? ArgLoader.model : ModelType.VGG16
                );
                break;
            case "CLASSIFY":
                String imageURI = ArgLoader.imageURI;
                String modelLocation = ArgLoader.modelLoc;

                List<String> labelNameList = new ArrayList();
                for (String labelName : ArgLoader.labelList) {
                    labelNameList.add(imageURI);
                }
                final List<List<String>> classify = api.classify(modelLocation, labelNameList, imageURI);
                for (List<String> result : classify) {
                    logger.info("Labels: " + result.toString());
                }

                break;
            default:

        }
        logger.info("Finnished");
    }

    private static Logger setupLogBack(ch.qos.logback.classic.Logger logbackLogger) {
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();

        FileAppender fileAppender = new FileAppender();
        fileAppender.setContext(loggerContext);
        fileAppender.setName("timestamp");
        // set the file name
        fileAppender.setFile(ArgLoader.logLocation + "log/" + System.currentTimeMillis() + ".log");

        PatternLayoutEncoder encoder = new PatternLayoutEncoder();
        encoder.setContext(loggerContext);
        encoder.setPattern("%r %thread %level - %msg%n");
        encoder.start();

        fileAppender.setEncoder(encoder);
        fileAppender.start();

        // attach the rolling file appender to the logger of your choice
        logbackLogger.addAppender(fileAppender);
        return logbackLogger;
    }

    private static void setCustomUncaughtExceptionHandler() {
        // Making sure that if one thread crashes,
        // then the whole JVM will shut down.
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            public void uncaughtException(Thread t, Throwable e) {
                System.out.println(t + " throws exception: " + e);
                System.exit(1);
            }
        });
    }

}
