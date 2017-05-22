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
import cz.muni.fi.imageNet.Pojo.NeuralNetModel;

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

        if (ArgLoader.learningRate != null) {
            config.setLearningRate(ArgLoader.learningRate);
        }

        ImageNetAPI api = new ImageNetAPI(config);

        switch (ArgLoader.method) {
            case "TRAIN":
                File datasetFile = new File(ArgLoader.datasetLoc);
                if (!datasetFile.isFile() || !datasetFile.canRead()) {
                    throw new IllegalArgumentException("There is wrong path to dataset file.");
                }
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
                    labelNameList.add(labelName);
                }
                final List<List<String>> classify = api.classify(modelLocation, labelNameList, imageURI);
                for (List<String> result : classify) {
                    logger.info("Labels: " + result.toString());
                }

                break;
            case "EVALUATE":
                File datasetFile1 = new File(ArgLoader.datasetLoc);
                if (!datasetFile1.isFile() || !datasetFile1.canRead()) {
                    throw new IllegalArgumentException("There is wrong path to dataset file.");
                }

                String modelLocation1 = ArgLoader.modelLoc;

                List<String> labelNameList1 = new ArrayList();
                for (String labelName : ArgLoader.labelList) {
                    labelNameList1.add(labelName);
                }

                List<DataSampleDTO> datasetList1 = new ArrayList();

                try (final BufferedReader fileReader = new BufferedReader(new FileReader(datasetFile1))) {
                    String line = fileReader.readLine();
                    while (line != null) {

                        DataSampleDTO sample = new DataSampleDTO(line);
                        datasetList1.add(sample);
                        line = fileReader.readLine();
                    }
                }

                DataSampleDTO[] dataset1 = new DataSampleDTO[datasetList1.size()];
                dataset1 = datasetList1.toArray(dataset1);
                final String evaluate = api.evaluateModel(
                        new File(modelLocation1), 
                        dataset1, 
                        labelNameList1,
                        ArgLoader.model != null ? ArgLoader.model : ModelType.VGG16);
                logger.info(evaluate);
                break;
            default:

        }
        logger.info("Finnished");
    }

}
