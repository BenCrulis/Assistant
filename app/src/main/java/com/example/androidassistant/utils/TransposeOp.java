package com.example.androidassistant.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PointF;
import android.util.Log;

import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class TransposeOp implements ImageOperator {
    @Override
    public TensorImage apply(TensorImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int totalNbPixels = width * height;

        Bitmap inBitmap = image.getBitmap();

        if (inBitmap == null) System.exit(1);

        Log.i("info", "total number of pixels: " + totalNbPixels + ", capacity: " + image.getBuffer().capacity());
        //assert totalNbPixels == inIntBuffer.capacity();

        ByteBuffer buffer = ByteBuffer.allocate(totalNbPixels*4);
        buffer.rewind();

        //Bitmap outBitmap = BitmapFactory.decodeByteArray(buffer.array(), 0, totalNbPixels*4);
        Bitmap outBitmap = Bitmap.createBitmap(height, width, inBitmap.getConfig());
        //Bitmap outBitmap = inBitmap;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                //Log.i("pixel", "x: " + x + ", y: " + y);
                int color = inBitmap.getPixel(x, y);
                outBitmap.setPixel(y, x, color);
            }
        }

        TensorImage outImage = TensorImage.fromBitmap(outBitmap);
        Log.i("transpose", "done.");

        return outImage;
    }

    @Override
    public int getOutputImageWidth(int inputImageHeight, int inputImageWidth) {
        return inputImageHeight;
    }

    @Override
    public int getOutputImageHeight(int inputImageHeight, int inputImageWidth) {
        return inputImageWidth;
    }

    @Override
    public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
        return new PointF(point.y, point.x);
    }
}
