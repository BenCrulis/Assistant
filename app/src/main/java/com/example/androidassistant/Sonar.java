package com.example.androidassistant;

import static com.example.androidassistant.utils.Image.getImagenetNormalizeOp;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import com.example.androidassistant.utils.Image;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;

public class Sonar {

    public static int DEPTH_IMAGE_SIZE = 518;

    private static final ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(DEPTH_IMAGE_SIZE, DEPTH_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    .add(getImagenetNormalizeOp())
                    .build();


    public static float getDistance(@NonNull ImageProxy image, Interpreter interpreter) {
        int height = image.getHeight();
        int width = image.getWidth();
        int planes = image.getPlanes().length;
        int imageRot = image.getImageInfo().getRotationDegrees();

        Log.i("PHOTO", "width: " + width + ", height: " + height);
        Log.i("PHOTO", "rotation: " + imageRot + "°");
        Bitmap bitmap = image.toBitmap();
        bitmap = bitmap.copy(bitmap.getConfig(), false);
//                        queueSpeak("L'image est de taille " + width + " par " + height +
//                                " pixels, avec " + planes + " plans");
        IntBuffer buffer = image.getPlanes()[0]
                .getBuffer()
                .asIntBuffer()
                .asReadOnlyBuffer();
        buffer.rewind();
        int[] intArray = new int[buffer.remaining()];
        buffer.get(intArray);
//                Log.i("buffer", "toString: " + Arrays.toString(intArray));

        // Create a TensorImage object. This creates the tensor of the corresponding
        // tensor type that the TensorFlow Lite interpreter needs.
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        // Analysis code for every frame
        // Preprocess the image
        tensorImage.load(bitmap);

        // depth prediction
        long beforeDepth = System.currentTimeMillis();

        tensorImage = imageProcessor.process(tensorImage);
        long elapsedDepth = System.currentTimeMillis() - beforeDepth;
        Log.i("depth", "depth preprocessing: " + elapsedDepth + " ms");

        int depthWidth = tensorImage.getWidth();
        int depthHeight = tensorImage.getHeight();
        FloatBuffer depthOutput = FloatBuffer.allocate(depthHeight * depthWidth);

        beforeDepth = System.currentTimeMillis();
        interpreter.run(tensorImage.getBuffer().asFloatBuffer(), depthOutput);
        elapsedDepth = System.currentTimeMillis() - beforeDepth;
        Log.i("depth", "depth inference: " + elapsedDepth + " ms");

        // get depth value in the middle of the image

        float depthValue = 0.0f;
        int middleIndex = depthWidth * depthHeight / 2;
        depthOutput.rewind();
        depthOutput.position(middleIndex);
        depthValue = depthOutput.get();
        Log.i("depth", "depth value: " + depthValue);

//        depthOutput = Image.rgbImageTranspose(depthOutput, DEPTH_IMAGE_SIZE, DEPTH_IMAGE_SIZE);

//        Bitmap finalDepthBitmap = Image.greyscaleBufferToBitmap(depthOutput, depthWidth, depthHeight);
//        Log.i("BITMAP", Image.describeBitmap(finalDepthBitmap));
        //Bitmap finalDepthBitmap = bitmap; // works
//        activity.runOnUiThread(() -> {
//            Log.i("BITMAP", "displaying depth bitmap");
//            displayPhoto(finalDepthBitmap, imageRot);
//            Log.i("BITMAP", "issued command: displaying depth bitmap");
//        });

        return depthValue;
    }

}
