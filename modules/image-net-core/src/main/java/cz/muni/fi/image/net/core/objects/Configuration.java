package cz.muni.fi.image.net.core.objects;

import java.util.Random;

/**
 * @author Jakub Peschel
 */
public class Configuration {

    //<editor-fold defaultstate="collapsed" desc="imageDownloadFolder">
    /**
     * Location used for storing images. By default system use linux "/tmp/imageNet".
     */
    private String imageDownloadFolder = "/tmp/imageNet";

    /**
     * Getter for {@link Configuration#imageDownloadFolder} constant.
     *
     * @return Location in String
     */
    public String getImageDownloadFolder() {
        return imageDownloadFolder;
    }

    /**
     * Setter for {@link Configuration#imageDownloadFolder} constant.
     *
     * @param imageDownloadFolder
     */
    public void setImageDownloadFolder(String imageDownloadFolder) {
        this.imageDownloadFolder = imageDownloadFolder;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="tempFolder">
    /**
     * Location used for storing models. By default system use linux "/tmp/imageNet".
     */
    private String tempFolder = "/tmp/imageNet";

    /**
     * Getter for {@link Configuration#tempFolder} constant.
     *
     * @return Location in String
     */
    public String getTempFolder() {
        return tempFolder;
    }

    /**
     * Setter for {@link Configuration#tempFolder} constant.
     *
     * @param tempFolder
     */
    public void setTempFolder(String tempFolder) {
        this.tempFolder = tempFolder;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="corePoolSize">
    /**
     * Number of threads
     */
    private int corePoolSize = 10;

    public int getCorePoolSize() {
        return corePoolSize;
    }

    public void setCorePoolSize(int corePoolSize) {
        this.corePoolSize = corePoolSize;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="maximumPoolSize">
    /**
     *
     */
    private int maximumPoolSize = 10;

    public int getMaximumPoolSize() {
        return maximumPoolSize;
    }

    public void setMaximumPoolSize(int maximumPoolSize) {
        this.maximumPoolSize = maximumPoolSize;
    }

    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="keepAliveTime">
    /**
     *
     */
    private int keepAliveTime = 10;

    public int getKeepAliveTime() {
        return keepAliveTime;
    }

    public void setKeepAliveTime(int keepAliveTime) {
        this.keepAliveTime = keepAliveTime;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="epoch">
    /**
     *
     */
    private int epoch = 10000;

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="seed">
    /**
     *
     */
    private int seed = new Random().nextInt();

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="iterations">
    /**
     *
     */
    private int iterations = 4;

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="learningRate">
    /**
     *
     */
    private double learningRate = 0.1;

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="failedTreshold">
    /**
     *
     */
    private int failedTreshold = 1000;

    public int getFailedTreshold() {
        return failedTreshold;
    }

    public void setFailedTreshold(int failedTreshold) {
        this.failedTreshold = failedTreshold;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="isTimed">
    /**
     *
     */
    private boolean timed = false;

    public boolean isTimed() {
        return timed;
    }

    public void setTimed(boolean timed) {
        this.timed = timed;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="time">
    /**
     *
     */
    private long time = 0;

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="gpuCount">
    /**
     *
     */
    private int GPUCount = 1;

    public int getGPUCount() {
        return GPUCount;
    }

    public void setGPUCount(int GPUCount) {
        this.GPUCount = GPUCount;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="batchSize">
    /**
     *
     */
    private int batchSize = 32;

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="DropOut">
    /**
     *
     */
    private double dropout = 0;

    public double getDropout() {
        return dropout;
    }

    public void setDropout(double dropout) {
        this.dropout = dropout;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="L1">
    /**
     *
     */
    private double l1 = 0;

    public double getL1() {
        return l1;
    }

    public void setL1(double l1) {
        this.l1 = l1;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="L2">
    /**
     *
     */
    private double l2 = 0;

    public double getL2() {
        return l2;
    }

    public void setL2(double l2) {
        this.l2 = l2;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Momentum">
    /**
     *
     */
    private double momentum = 0;

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
    //</editor-fold>
}
