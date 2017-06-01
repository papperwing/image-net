package cz.muni.fi.imagenet.visualizer;

import java.io.File;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;

/**
 *
 * @author jpeschel
 */
public class Visualizer {

    public static void main(String[] args) {
        UIServer uiServer = UIServer.getInstance();
        File storageFile = new File("./storage_file");
        StatsStorage storage = new J7FileStatsStorage(storageFile);
        uiServer.attach(storage);
    }

}
