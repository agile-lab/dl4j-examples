package org.deeplearning4j.examples.deeplogo;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Created by paolo on 18/05/2017.
 */
public class MyNativeImageLoader extends NativeImageLoader {

    public MyNativeImageLoader(int height, int width, int channels, ImageTransform imageTransform, boolean centerCropIfNeeded) {
        super(height, width, channels, imageTransform);
        this.centerCropIfNeeded = centerCropIfNeeded;
    }

    @Override
    protected opencv_core.Mat centerCropIfNeeded(opencv_core.Mat img) {
        int x = 0;
        int y = 0;
        int height = img.rows();
        int width = img.cols();
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - 2*diff;
        } else if (height > width) {
            y = diff;
            height = height - 2*diff;
        }

        return img.apply(new opencv_core.Rect(x, y, width, height));
    }

    @Override
    protected opencv_core.Mat scalingIfNeed(opencv_core.Mat image, int dstHeight, int dstWidth) {
        opencv_core.Mat scaled = image;
        if (dstHeight > 0 && dstWidth > 0 && (image.rows() != dstHeight || image.cols() != dstWidth)) {
            resize(image, scaled = new opencv_core.Mat(), new opencv_core.Size(dstWidth, dstHeight));
        }

        return scaled;
    }
}
