package com.example.androidassistant.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class Image {
    static float[] MEAN_RGB = {0.485f * 255, 0.456f * 255, 0.406f * 255};
    static float[] STDDEV_RGB = {0.229f * 255, 0.224f * 255, 0.225f * 255};


    public static NormalizeOp getImagenetNormalizeOp() {
        return new NormalizeOp(MEAN_RGB, STDDEV_RGB);
    }

    public static NormalizeOp getIntNormalizeOp() {
        return new NormalizeOp(new float[]{0.0f, 0.0f, 0.0f}, new float[]{255.0f, 255.0f, 255.0f});
    }


    public static TensorImage rotate90(TensorImage tensorImage, int k) {
        ImageProcessor rotate90processor =
            new ImageProcessor.Builder()
                .add(new Rot90Op(k))
                .build();
        return rotate90processor.process(tensorImage);
    }

    public static int compute_k_from_degrees(int rot) {
        if (rot < 0) {
            throw new IllegalArgumentException("rotation cannot be negative");
        } else if (rot % 90 != 0) {
            throw new IllegalArgumentException("rotation should be a positive multiple of 90");
        }
        return rot / 90;
    }

    public static FloatBuffer rgbImageTranspose(FloatBuffer in, int width, int height) {
        FloatBuffer out = FloatBuffer.allocate(in.capacity());
        out.rewind();
        float[] in_array = in.array();

        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                int pos = (i + j*width);
                out.put(in_array[pos]);
            }
        }
        out.rewind();
        return out;
    }

    public static String describeBitmap(Bitmap bitmap) {
        String out = "";
        out += "has alpha: " + bitmap.hasAlpha() + ", ";
        out += "has mipMap: " + bitmap.hasMipMap() + ", ";
        out += "config: " + bitmap.getConfig() + ", ";
        out += "width: " + bitmap.getWidth() + ", ";
        out += "height: " + bitmap.getHeight() + ", ";
        out += "density: " + bitmap.getDensity() + ", ";
        out += "color space: " + bitmap.getColorSpace() + ", ";
        out += "row bytes: " + bitmap.getRowBytes() + ", ";
        out += "byte count: " + bitmap.getByteCount() + ", ";
        out += "alloc byte count: " + bitmap.getAllocationByteCount() + ", ";
        out += "gen id: " + bitmap.getGenerationId();
        return out;
    }

    public static Bitmap greyscaleBufferToBitmap(FloatBuffer buffer, int width, int height) {
        buffer.rewind();

        float maxVal = 0.0f;
        float[] arr = buffer.array();
        for (float v : arr) {
            maxVal = Math.max(maxVal, v);
        }

        buffer.rewind();

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setHasAlpha(false);
        //if (true) return bitmap;
        //bitmap.copyPixelsFromBuffer(buffer);

        int totalNbPixels = width*height;

        ByteBuffer byteBuffer = ByteBuffer.allocate(width*height*4);
        byteBuffer.rewind();

        for (int i = 0; i < arr.length; i++) {
            int greyScaleVal = Math.round((arr[i] / maxVal) * 255);
            assert greyScaleVal < 256;
            int color = (0xff) << 24 | (greyScaleVal & 0xff) << 16 | (greyScaleVal & 0xff) << 8 | (greyScaleVal & 0xff);
            byteBuffer.put((byte) greyScaleVal);
            byteBuffer.put((byte) greyScaleVal);
            byteBuffer.put((byte) greyScaleVal);
            byteBuffer.put((byte) 255);

            //byteBuffer.put((byte) 255);byteBuffer.put((byte) 255);byteBuffer.put((byte) 255);byteBuffer.put((byte) 255);
        }
        Log.i("buffer", "current position: " + byteBuffer.position());
        byteBuffer.rewind();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        options.outConfig = Bitmap.Config.ARGB_8888;
        options.outWidth = width;
        options.outHeight = height;
        bitmap.copyPixelsFromBuffer(byteBuffer);
        //bitmap = BitmapFactory.decodeByteArray(byteBuffer.array(), 0, totalNbPixels*4, options);
        assert bitmap != null;
        //bitmap.setHasAlpha(false);
        //bitmap.setWidth(width);
        //bitmap.setHeight(height);
        //bitmap.copyPixelsFromBuffer(byteBuffer.asIntBuffer());

        return bitmap;
    }


}
