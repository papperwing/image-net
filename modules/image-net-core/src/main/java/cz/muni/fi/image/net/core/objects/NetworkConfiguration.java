package cz.muni.fi.image.net.core.objects;


import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

/**
 * Store for network configuration.
 *
 * 
 * @author Jakub Peschel
 */
public class NetworkConfiguration {
    private final ComputationGraphConfiguration config;

    public NetworkConfiguration(ComputationGraphConfiguration config) {
        this.config = config;
    }

    public ComputationGraphConfiguration getConfiguration() {
        return config;
    }
    
}
