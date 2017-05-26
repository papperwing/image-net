package cz.muni.fi.imageNet.downloader.enums;

/**
 * Enum for return state of image download
 *
 * @author Jakub Peschel
 */
public enum DownloadState {

    DOWNLOADED,
    FAILED,
    TIMED_OUT,
    ON_DISK
}
