package com.example.androidassistant.utils;

import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;


public class Image {
    static float[] MEAN_RGB = {0.485f * 255, 0.456f * 255, 0.406f * 255};
    static float[] STDDEV_RGB = {0.229f * 255, 0.224f * 255, 0.225f * 255};


    public static NormalizeOp getImagenetNormalizeOp() {
        return new NormalizeOp(MEAN_RGB, STDDEV_RGB);
    }


}
