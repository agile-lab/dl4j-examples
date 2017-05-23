package org.deeplearning4j.examples.deeplogo;

/**
 * Created by andlatel on 22/05/2017.
 */
/*-
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Rotates and scales images deterministically or randomly. Calls
 * {@link org.bytedeco.javacpp.opencv_imgproc#warpAffine(opencv_core.Mat, opencv_core.Mat, opencv_core.Mat, opencv_core.Size, int, int, opencv_core.Scalar)}
 * with given properties (interMode, borderMode, and borderValue).
 *
 * @author saudet
 */
public class MyRotateImageTransform extends BaseImageTransform<opencv_core.Mat> {

    float angle, scale;
    int interMode = INTER_LINEAR;
    int borderMode = BORDER_CONSTANT;
    opencv_core.Scalar borderValue = opencv_core.Scalar.ZERO;

    public MyRotateImageTransform(Random random, float angle, float scale) {
        super(random);
        this.angle = angle;
        this.scale = scale;
        converter = new OpenCVFrameConverter.ToMat();
    }


    @Override
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }

        opencv_core.Mat mat = converter.convert(image.getFrame());

        float cy = mat.rows() / 2 ;
        float cx = mat.cols() / 2 ;
        opencv_core.Mat M = getRotationMatrix2D(new opencv_core.Point2f(cx, cy), angle, scale) ;

        opencv_core.Mat result = new opencv_core.Mat();
        warpAffine(mat, result, M, mat.size());

        return new ImageWritable(converter.convert(result));
    }

}
