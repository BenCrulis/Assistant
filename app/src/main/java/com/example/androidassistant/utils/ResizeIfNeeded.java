package com.example.androidassistant.utils;

import android.graphics.PointF;

import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

public class ResizeIfNeeded implements ImageOperator {
    private int targetWidth;
    private int targetHeight;
    private ResizeOp resizeOp;

    public ResizeIfNeeded(int targetHeight, int targetWidth) {
        this.targetHeight = targetHeight;
        this.targetWidth = targetWidth;
        this.resizeOp = new ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR);
    }


    @Override
    public TensorImage apply(TensorImage image) {
        if (image.getHeight() == this.targetHeight && image.getWidth() == this.targetWidth) {
            return image;
        }

        return this.resizeOp.apply(image);
    }

    @Override
    public int getOutputImageWidth(int inputImageHeight, int inputImageWidth) {
        return this.targetWidth;
    }

    @Override
    public int getOutputImageHeight(int inputImageHeight, int inputImageWidth) {
        return this.targetHeight;
    }

    @Override
    public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
        return this.resizeOp.inverseTransform(point, inputImageHeight, inputImageWidth);
    }
}
