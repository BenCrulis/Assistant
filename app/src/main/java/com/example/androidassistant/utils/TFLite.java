package com.example.androidassistant.utils;

import android.app.Activity;
import android.util.Log;

import androidx.camera.core.internal.ByteBufferOutputStream;

import com.example.androidassistant.utils.alloc.NativeByteBuffer;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

public class TFLite {

    public static class ModelWithBuffer {
        public final Interpreter interpreter;
        public final ByteBuffer nativeBuffer;

        public ModelWithBuffer(Interpreter interpreter, ByteBuffer nativeBuffer) {
            this.interpreter = interpreter;
            this.nativeBuffer = nativeBuffer;
        }

        public static ModelWithBuffer modelWithBuffer(Interpreter interpreter, ByteBuffer nativeBuffer) {
            return new ModelWithBuffer(interpreter, nativeBuffer);
        }

        public void unsafeDeallocate() {
            NativeByteBuffer.deallocateBuffer(this.nativeBuffer);
        }
    }

    public static ByteBuffer createBuffer(int size) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(size);
        //ByteBuffer buffer = NativeByteBuffer.createBuffer(size);
        return buffer;
    }

    public static Interpreter loadModel(InputStream inputStream, Interpreter.Options options) {
        int k = 0;
        try {
            byte[] buffer = new byte[65536];
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            k = inputStream.read(buffer);

            while (k > 0) {
                outputStream.write(buffer, 0, k);
                k = inputStream.read(buffer);
            }
            Log.i("bytebuffer", "k: " + k);
            outputStream.flush();
            byte[] modelBytes = outputStream.toByteArray();
            inputStream.close();
            outputStream.close();

            Log.i("bytebuffer", "byte[] array size: " + modelBytes.length);
            ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
            Log.i("bytebuffer", "bytebuffer size: " + modelBuffer.capacity());

            modelBuffer.put(modelBytes);
            modelBuffer.rewind();


            Log.i("bytebuffer", "is direct: " + modelBuffer.isDirect());

            Interpreter model = new Interpreter(modelBuffer, options);
            return model;
        }
        catch (Exception e) {
            Log.e("TFLite utils", "exception", e);
            Log.e("TFLite utils", "k was " + k);
            System.exit(1);
            return null;
        }
    }


    public static Interpreter loadModelWithSize(InputStream inputStream, int size, Interpreter.Options options) {
        int k = 0;
        try {
            byte[] buffer = new byte[65536];
            k = inputStream.read(buffer);

            ByteBuffer modelBuffer = createBuffer(size);

            while (k > 0) {
                modelBuffer.put(buffer, 0, k);
                k = inputStream.read(buffer);
            }
            Log.i("bytebuffer", "k: " + k);
            inputStream.close();
            modelBuffer.rewind();

            Log.i("bytebuffer", "is direct: " + modelBuffer.isDirect());

            Interpreter model = new Interpreter(modelBuffer, options);
            return model;
        }
        catch (Exception e) {
            Log.e("TFLite utils", "exception", e);
            Log.e("TFLite utils", "k was " + k);
            System.exit(1);
            return null;
        }
    }

    /**
     * This method allocates native memory and returns it alongside the Interpreter object.
     * /!\ The ByteBuffer does not automatically deallocate the native memory /!\
     * @param inputStream
     * @param size
     * @param options
     * @return
     */
    public static ModelWithBuffer loadModelWithSizeUnsafe(InputStream inputStream, int size, Interpreter.Options options) {
        Log.i("native", "Attempting model loading with native ByteBuffer");
        int k = 0;
        try {
            byte[] buffer = new byte[65536];
            k = inputStream.read(buffer);

            ByteBuffer modelBuffer = NativeByteBuffer.createBuffer(size);

            while (k > 0) {
                modelBuffer.put(buffer, 0, k);
                k = inputStream.read(buffer);
            }
            Log.i("bytebuffer", "k: " + k);
            inputStream.close();
            modelBuffer.rewind();

            Log.i("bytebuffer", "is direct: " + modelBuffer.isDirect());

            Interpreter model = new Interpreter(modelBuffer, options);
            return new ModelWithBuffer(model, modelBuffer);
        }
        catch (Exception e) {
            Log.e("TFLite utils", "exception", e);
            Log.e("TFLite utils", "k was " + k);
            System.exit(1);
            return null;
        }
    }

    public static ModelWithBuffer loadModelWithSizeUnsafe(InputStream inputStream, int size) {
        Interpreter.Options options = new Interpreter.Options()
                .setNumThreads(4)
                .setCancellable(true);
        return loadModelWithSizeUnsafe(inputStream, size, options);
    }

    public static Interpreter.Options getDefaultOptions() {
        return new Interpreter.Options()
                .setNumThreads(1)
                .setCancellable(true);
    }
}
