package com.example.androidassistant.utils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class TensorUtils {

    public static TensorBufferFloat parseStringVector(String vector) {
        String[] parts = vector.split(",");
        float[] floats = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            floats[i] = Float.parseFloat(parts[i].trim());
        }
        return FloatTensorFromFloats(floats);
    }


    public static TensorBufferFloat readVectorFromFile(String path) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(path));
            // get first line
            String firstLine = lines.get(0);

            return parseStringVector(firstLine);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public static TensorBufferFloat FloatTensorFromBuffer(ByteBuffer buffer) {
        TensorBufferFloat tensorBufferFloat = (TensorBufferFloat) TensorBufferFloat.createDynamic(DataType.FLOAT32);
        tensorBufferFloat.loadBuffer(buffer);
        return tensorBufferFloat;
    }

    public static TensorBufferFloat FloatTensorFromBytes(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        return FloatTensorFromBuffer(buffer);
    }


    public static TensorBufferFloat FloatTensorFromFloats(float[] floats) {
        TensorBufferFloat tensorBufferFloat = (TensorBufferFloat) TensorBufferFloat
                .createFixedSize(new int[]{floats.length}, DataType.FLOAT32);
        tensorBufferFloat.loadArray(floats);
        return tensorBufferFloat;
    }

    public static TensorBufferFloat FloatTensorFromFloatValues(float... floats) {
        TensorBufferFloat tensorBufferFloat = (TensorBufferFloat) TensorBufferFloat
                .createFixedSize(new int[]{floats.length}, DataType.FLOAT32);
        tensorBufferFloat.loadArray(floats);
        return tensorBufferFloat;
    }


    public static TensorBufferFloat addFloatTensors(List<TensorBufferFloat> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Tensor list is empty");
        }

        TensorBufferFloat firstTensor = tensors.get(0);
        int[] shape = firstTensor.getShape();
        int size = 1;
        for (int d : shape) {
            size *= d;
        }


        float[] accum = firstTensor.getFloatArray();

        for (int i = 1; i < tensors.size(); i++) {
            TensorBufferFloat current = tensors.get(i);
            int[] currentShape = current.getShape();
            if (!Arrays.equals(shape, currentShape)) {
                throw new IllegalArgumentException("Tensor at index " + i + " is of a different shape");
            }

            float[] currentValues = current.getFloatArray();

            for (int j = 0; j < size ; j++) {
                accum[j] += currentValues[j];
            }

        }

        TensorBufferFloat out = zeroLike(firstTensor);
        out.loadArray(accum);

        return out;
    }

    public static TensorBufferFloat zeros(int size) {
        TensorBufferFloat out = (TensorBufferFloat) TensorBufferFloat
                .createFixedSize(new int[]{size}, DataType.FLOAT32);
        return out;
    }

    public static TensorBufferFloat zeroLike(TensorBufferFloat tensorBufferFloat) {
        TensorBufferFloat out = (TensorBufferFloat) TensorBufferFloat
                .createFixedSize(tensorBufferFloat.getShape(), DataType.FLOAT32);
        return out;
    }

    public static TensorBufferFloat flatConcat(List<TensorBufferFloat> tensorList) {
        int size = 0;
        for (TensorBufferFloat tensorBufferFloat : tensorList) {
            size += tensorBufferFloat.getFlatSize();
        }

        float[] floats = new float[size];

        int i = 0;
        for (TensorBufferFloat tensorBufferFloat : tensorList) {
            float[] arr = tensorBufferFloat.getFloatArray();
            System.arraycopy(arr, 0, floats, i, arr.length);
            i += tensorBufferFloat.getFlatSize();
        }

        TensorBufferFloat out = zeros(size);
        out.loadArray(floats);

        return out;
    }

    public static TensorBufferFloat divideTensorByNumber(TensorBufferFloat tensorBufferFloat, float divisor) {
        float[] floats = tensorBufferFloat.getFloatArray();

        for (int i = 0; i < floats.length; i++) {
            floats[i] /= divisor;
        }

        TensorBufferFloat out = zeroLike(tensorBufferFloat);
        out.loadArray(floats);

        return out;
    }

    public static TensorBufferFloat normalizeVector(TensorBufferFloat vector) {
        float[] floats = vector.getFloatArray();

        double norm_sq = 0.0;

        for (float x: floats) {
            norm_sq += x*x;
        }

        return divideTensorByNumber(vector, (float) Math.sqrt(norm_sq));
    }

    public static double dot(TensorBufferFloat a, TensorBufferFloat b) {
        float[] a_arr = a.getFloatArray();
        float[] b_arr = b.getFloatArray();
        double out = 0.0;
        for (int i = 0; i < a_arr.length; i++) {
            out += a_arr[i] * b_arr[i];
        }
        return out;
    }

    public static TensorBufferFloat softmax(TensorBufferFloat tensor) {
        float[] floats = tensor.getFloatArray();
        float max = Float.MIN_VALUE;
        for (float x: floats) {
            if (x > max) {
                max = x;
            }
        }

        for (int i = 0; i < floats.length; i++) {
            float currentVal = (float) Math.exp(floats[i] - max);
            floats[i] = currentVal;
        }

        float sum = 0.0f;
        for (float x: floats) {
            sum += x;
        }

        for (int i = 0; i < floats.length; i++) {
            floats[i] /= sum;
        }

        TensorBufferFloat out = zeroLike(tensor);
        out.loadArray(floats);

        return out;
    }


    public static float reduceAdd(TensorBufferFloat tensor) {
        float[] floats = tensor.getFloatArray();
        float out = 0.0f;
        for (float x: floats) {
            out += x;
        }
        return out;
    }

    public static TensorBufferFloat map(TensorBufferFloat tensor, Function<Float, Float> func) {
        float[] floats = tensor.getFloatArray();
        for (int i = 0; i < floats.length; i++) {
            floats[i] = func.apply(floats[i]);
        }
        TensorBufferFloat out = zeroLike(tensor);
        out.loadArray(floats);
        return out;
    }

    public static TensorBufferFloat normalize(TensorBufferFloat tensor) {
        float sumSquared = reduceAdd(map(tensor, x -> x*x));
        return map(tensor, x -> x / (float) Math.sqrt(sumSquared));
    }

}
