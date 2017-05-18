package cz.muni.fi.imageNet.manager;

import cz.muni.fi.imageNet.Pojo.Configuration;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author jpeschel
 */
public class INDASerializer {

    public static Logger logger = LoggerFactory.getLogger(INDASerializer.class);

    public static Configuration conf;

    public static File serializeINDA(INDArray array, String key) throws FileNotFoundException, IOException {
        if (INDASerializer.conf == null) {
            throw new IllegalStateException("configuration wasnt passed");
        }
        final File file = new File(INDASerializer.conf.getTempFolder() + key + ".bin");
        try (FileOutputStream stream = new FileOutputStream(
                file)) {
            try (ObjectOutputStream out = new ObjectOutputStream(stream)) {
                out.writeObject(array);
            }
        }
        return file;
    }

    public static INDArray deserializeINDA(String key) throws FileNotFoundException, IOException, ClassNotFoundException {
        if (INDASerializer.conf == null) {
            throw new IllegalStateException("configuration wasnt passed");
        }
        final File file = new File(INDASerializer.conf.getTempFolder() + key + ".bin");
        INDArray result = null;
        try (FileInputStream stream = new FileInputStream(file)) {
            try (ObjectInputStream out = new ObjectInputStream(stream)) {
                result = (INDArray) out.readObject();
            }
        }
        return result;
    }
}
