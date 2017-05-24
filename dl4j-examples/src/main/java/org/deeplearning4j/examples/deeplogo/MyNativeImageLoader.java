package org.deeplearning4j.examples.deeplogo;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.lept;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteOrder;

import static org.bytedeco.javacpp.lept.*;
import static org.bytedeco.javacpp.lept.pixConvert4To8;
import static org.bytedeco.javacpp.lept.pixDestroy;
import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_core.mixChannels;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_ANYCOLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_ANYDEPTH;
import static org.bytedeco.javacpp.opencv_imgcodecs.imdecode;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2BGR;

/**
 * Created by paolo on 18/05/2017.
 */
public class MyNativeImageLoader extends NativeImageLoader {

    public MyNativeImageLoader(int height, int width, int channels, ImageTransform imageTransform, boolean centerCropIfNeeded) {
        super(height, width, channels, imageTransform);
        this.centerCropIfNeeded = centerCropIfNeeded;
    }

    static opencv_core.Mat convert(lept.PIX pix) {
        lept.PIX tempPix = null;
        if (pix.colormap() != null) {
            lept.PIX pix2 = pixRemoveColormap(pix, REMOVE_CMAP_TO_FULL_COLOR);
            tempPix = pix = pix2;
        } else if (pix.d() < 8) {
            lept.PIX pix2 = null;
            switch (pix.d()) {
                case 1:
                    pix2 = pixConvert1To8(null, pix, (byte) 0, (byte) 255);
                    break;
                case 2:
                    pix2 = pixConvert2To8(pix, (byte) 0, (byte) 85, (byte) 170, (byte) 255, 0);
                    break;
                case 4:
                    pix2 = pixConvert4To8(pix, 0);
                    break;
                default:
                    assert false;
            }
            tempPix = pix = pix2;
        }
        int height = pix.h();
        int width = pix.w();
        int channels = pix.d() / 8;
        opencv_core.Mat mat = new opencv_core.Mat(height, width, CV_8UC(channels), pix.data(), 4 * pix.wpl());
        opencv_core.Mat mat2 = new opencv_core.Mat(height, width, CV_8UC(channels));
        // swap bytes if needed
        int[] swap = {0, 3, 1, 2, 2, 1, 3, 0}, copy = {0, 0, 1, 1, 2, 2, 3, 3},
            fromTo = channels > 1 && ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN) ? swap : copy;
        mixChannels(mat, 1, mat2, 1, fromTo, fromTo.length / 2);
        if (tempPix != null) {
            pixDestroy(tempPix);
        }
        return mat2;
    }


    public opencv_core.Mat asOpenCVMat(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            return toMat(bis);
        }
    }

    public opencv_core.Mat toMat(InputStream is) throws IOException {
        byte[] bytes = IOUtils.toByteArray(is);
        opencv_core.Mat image = imdecode(new opencv_core.Mat(bytes), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        if (image == null || image.empty()) {
            PIX pix = pixReadMem(bytes, bytes.length);
            if (pix == null) {
                throw new IOException("Could not decode image from input stream");
            }
            image = convert(pix);
            pixDestroy(pix);
        }
        return image;
    }

    public opencv_core.Mat crop(opencv_core.Mat image, int x, int y, int w, int h) throws IOException {

        return image.apply(new opencv_core.Rect(x, y, w, h));
    }


}
