package cz.muni.fi.imagenet.image.net.cli;

import cz.muni.fi.image.net.api.dto.DataSampleDTO;
import cz.muni.fi.image.net.core.objects.Configuration;
import cz.muni.fi.image.net.core.enums.ModelType;
import cz.muni.fi.image.net.api.ImageNetAPI;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sampullara.cli.Args;

import java.io.IOException;

/**
 * Simple console line interface for training and evaluation of neural networks.
 *
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class App {

    public static Logger logger = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) throws Exception {
        final List<String> parse;
        try {
            parse = Args.parse(ArgLoader.class, args);
        } catch (IllegalArgumentException e) {
            logger.error("Error:", e);
            Args.usage(ArgLoader.class);
            //System.exit(1);
            //return;
        }
        logger.info("Started command line interface.");

        Configuration config = new Configuration();

        //TODO: refactor passing commandline arguments into configuration
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

        if (ArgLoader.oLearningRate != null) {
            config.setOLearningRate(ArgLoader.oLearningRate);
        }

        if (ArgLoader.epochCount != null) {
            config.setEpoch(ArgLoader.epochCount);
        }

        if (ArgLoader.tempLoc != null) {
            config.setTempFolder(ArgLoader.tempLoc);
        }

        if (ArgLoader.gpuCount != null) {
            config.setGPUCount(ArgLoader.gpuCount);
        }

        if (ArgLoader.batchSize != null) {
            config.setBatchSize(ArgLoader.batchSize);
        }

        if (ArgLoader.l1 != null) {
            config.setL1(ArgLoader.l1);
        }

        if (ArgLoader.l2 != null) {
            config.setL2(ArgLoader.l2);
        }

        if (ArgLoader.ol2 != null) {
            config.setOutputL2(ArgLoader.ol2);
        }

        if (ArgLoader.dropout != null) {
            config.setDropout(ArgLoader.dropout);
        }

        if (ArgLoader.javaVersion != null) {
            config.setJavaMinorVersion(ArgLoader.javaVersion);
        }

        ImageNetAPI api = new ImageNetAPI(config);
        logger.info(config.toString());

        switch (ArgLoader.method) {
            case "TRAIN":
                train(api);
                break;
            case "CLASSIFY":
                classify(api);

                break;
            case "EVALUATE":
                evaluate(api);
                break;
            case "CONTINUE":
                continueTraining(api);
                break;
            default:

        }
        logger.info("Finnished");
    }

    private static void train(final ImageNetAPI api) throws IllegalArgumentException, IOException {
        final File datasetFile = new File(ArgLoader.datasetLoc);
        if (!datasetFile.isFile() || !datasetFile.canRead()) {
            throw new IllegalArgumentException("There is wrong path to dataset file.");
        }
        logger.info("Loading dataset: " + datasetFile.getAbsolutePath());

        final List<DataSampleDTO> datasetList = new ArrayList();

        try (final BufferedReader fileReader = new BufferedReader(new FileReader(datasetFile))) {
            String line = fileReader.readLine();
            while (line != null) {

                final DataSampleDTO sample = new DataSampleDTO(line);
                datasetList.add(sample);
                line = fileReader.readLine();
            }
        }

        DataSampleDTO[] dataset = new DataSampleDTO[datasetList.size()];
        dataset = datasetList.toArray(dataset);

        api.getModel(
                ArgLoader.modelName,
                dataset,
                ArgLoader.model != null ? ModelType.valueOf(ArgLoader.model) : ModelType.RESNET50
        );
    }

    private static void classify(final ImageNetAPI api) throws IOException {
        final String imageURI = ArgLoader.imageURI;
        final String modelLocation = ArgLoader.modelLoc;

        final List<String> labelNameList = new ArrayList();
        for (final String labelName : ArgLoader.labelList) {
            labelNameList.add(labelName);
        }
        //TODO: create object for List<List<>> object
        final List<List<String>> classify = api.classify(modelLocation, labelNameList, imageURI);
        for (List<String> result : classify) {
            logger.info("Labels: " + result.toString());
        }
    }

    private static void evaluate(final ImageNetAPI api) throws IllegalArgumentException, IOException {
        final File datasetFile1 = new File(ArgLoader.datasetLoc);
        if (!datasetFile1.isFile() || !datasetFile1.canRead()) {
            throw new IllegalArgumentException("There is wrong path to dataset file.");
        }

        final String modelLocation1 = ArgLoader.modelLoc;

        final List<String> labelNameList1 = new ArrayList();
        for (final String labelName : ArgLoader.labelList) {
            labelNameList1.add(labelName);
        }

        final List<DataSampleDTO> datasetList1 = new ArrayList();

        //TODO: refactor duplicate code
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
                ArgLoader.model != null ? ModelType.valueOf(ArgLoader.model) : ModelType.RESNET50);
        logger.info("\n" + evaluate);
    }

    private static void continueTraining(final ImageNetAPI api) throws IllegalArgumentException, IOException {
        final File datasetFile = new File(ArgLoader.datasetLoc);
        if (!datasetFile.isFile() || !datasetFile.canRead()) {
            throw new IllegalArgumentException("There is wrong path to dataset file.");
        }
        logger.info("Loading dataset: " + datasetFile.getAbsolutePath());

        final List<DataSampleDTO> datasetList = new ArrayList();

        try (final BufferedReader fileReader = new BufferedReader(new FileReader(datasetFile))) {
            String line = fileReader.readLine();
            while (line != null) {

                final DataSampleDTO sample = new DataSampleDTO(line);
                datasetList.add(sample);
                line = fileReader.readLine();
            }
        }

        DataSampleDTO[] dataset = new DataSampleDTO[datasetList.size()];
        dataset = datasetList.toArray(dataset);


        final String modelLocation1 = ArgLoader.modelLoc;

        api.continueTraining(
                new File(modelLocation1),
                dataset,
                ArgLoader.model != null ? ModelType.valueOf(ArgLoader.model) : ModelType.RESNET50
        );
    }

}
