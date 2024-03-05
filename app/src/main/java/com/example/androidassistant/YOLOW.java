package com.example.androidassistant;

import static com.example.androidassistant.utils.Image.getImagenetNormalizeOp;
import static com.example.androidassistant.utils.Image.getIntNormalizeOp;

import android.app.Activity;
import android.app.Application;
import android.util.Log;

import com.example.androidassistant.utils.ArrayUtils;
import com.example.androidassistant.utils.DetectUtils;
import com.example.androidassistant.utils.TFLite;

import org.jetbrains.kotlinx.multik.api.ConstructorsKt;
import org.jetbrains.kotlinx.multik.api.Engine;
import org.jetbrains.kotlinx.multik.api.EngineType;
import org.jetbrains.kotlinx.multik.api.KEEngineType;
import org.jetbrains.kotlinx.multik.api.Multik;
import org.jetbrains.kotlinx.multik.api.EngineKt;
import org.jetbrains.kotlinx.multik.api.MultikKt;
import org.jetbrains.kotlinx.multik.api.NativeEngineType;
import org.jetbrains.kotlinx.multik.api.math._mathKt;
import org.jetbrains.kotlinx.multik.api.math.Math;
import org.jetbrains.kotlinx.multik.api.math.MathEx;
import org.jetbrains.kotlinx.multik.ndarray.data.D1;
import org.jetbrains.kotlinx.multik.ndarray.data.D2;
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension;
import org.jetbrains.kotlinx.multik.ndarray.data.DimensionsKt;
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray;
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray;
import org.jetbrains.kotlinx.multik.ndarray.data.NDArrayKt;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringJoiner;

import com.example.androidassistant.utils.TFLite;
import com.example.androidassistant.utils.TransposeOp;

public class YOLOW {
    public static int EMB_DIM = 512;
    public static int N_ANCHORS = 9261;

    public static final Engine engine = EngineKt.enginesProvider().get(KEEngineType.INSTANCE);
    public static final Math math = YOLOW.engine.getMath();
    public static final ImageProcessor yoloWProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(672, 672, ResizeOp.ResizeMethod.BILINEAR))
//                    .add(new ResizeWithCropOrPadOp(672, 672))
//                    .add(getImagenetNormalizeOp())
//                    .add(new TransposeOp())
                    .add(getIntNormalizeOp())
                    .build();

    private String model_size;
    private Interpreter model = null;
    private HashMap<String, YOLOWTargets> yolow_targets = null;


    static class YOLOWTargets {
        public TensorBuffer tensor;
        public String[] classes;

        public YOLOWTargets(TensorBuffer tensor, String[] classes) {
            this.tensor = tensor;
            this.classes = classes;
        }
    }

    public YOLOW(String model_size) {
        this.model_size = model_size;
        this.yolow_targets = new HashMap<>();
    }

    public static YOLOWTargets load_yolow_classes_file(Application app, String filename) {
        try (InputStream inputStream = app.getAssets().open(filename)) {
            BufferedReader bufferedReader = new BufferedReader(
                    new InputStreamReader(inputStream, StandardCharsets.UTF_8));
            String[] lines = bufferedReader.lines().toArray(String[]::new);
            Log.i("YOLOW", "number of lines in YOLOW classes file: " + lines.length);

            TensorBuffer tensor = TensorBufferFloat.createFixedSize(new int[]{lines.length, EMB_DIM}, DataType.FLOAT32);

            float[] floats = tensor.getFloatArray();

            String[] classes = new String[lines.length];

            for (int i = 0; i < lines.length; i++) {
                String line = lines[i];
                String[] cells = line.split(",");
                String class_name = cells[1].trim();
                if (class_name.isEmpty()) {
                    class_name = cells[0].trim();
                }
                classes[i] = class_name;
                for (int j = 0; j < cells.length - 2; j++) {
                    int loc = i*EMB_DIM + j;
                    float val = Float.parseFloat(cells[j+2].trim());
                    floats[loc] = val;
                }
            }
            tensor.loadArray(floats);
            return new YOLOWTargets(tensor, classes);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public void loadYOLOW(Application app, Interpreter.Options options) {
        String filename = "yolov8" + this.model_size + "-world.tflite";
        try {
            InputStream inputStream = app.getAssets().open(filename);
            int size = inputStream.available();

            model = TFLite.loadModelWithSize(inputStream, size, options);
            inputStream.close();
        }
        catch (IOException e) {
            Log.e("YOLOW", "exception", e);
            System.exit(1);
        }
    }

    public void add_targets(YOLOWTargets targets, String use_case) {
        this.yolow_targets.put(use_case, targets);
    }

    public YOLOWTargets get_targets(String use_case) {
        return yolow_targets.get(use_case);
    }


    public static ArrayList<Integer> classes_from_tensor(FloatBuffer tensor, double threshold) {
        ArrayList<Integer> objects = new ArrayList<>();
        tensor.rewind();
        float[] array = tensor.array();

        int n_classes = array.length / N_ANCHORS - 4;


        NDArray<Float, D2> ndarr = ConstructorsKt.ndarray(Multik.INSTANCE, array, N_ANCHORS, n_classes + 4);

        NDArray<Float, D1> max = math.maxD2(ndarr, 0);

        Log.i("YOLOW", "max: " + max);

        for (int i = 0; i < N_ANCHORS; i++) {

            float[] slice = new float[n_classes];
            for (int j = 0; j < n_classes; j++) {
                int idx = (n_classes + 4) * i + j + 4;
                slice[j] = array[idx];
            }

            int argmax = ArrayUtils.argmax(slice);
            float val = slice[argmax];
            if (val >= threshold) {
                objects.add(argmax);
            }
        }

        return objects;
    }

    public ArrayList<DetectUtils.Detect> infer(TensorImage image, String use_case, double IoUThresh, double min_prob) {
        YOLOWTargets targets = get_targets(use_case);
        if (targets == null) {
            Log.e("YOLOW", "unknown use case: " + use_case);
            System.exit(1);
        }

        TensorImage processed = yoloWProcessor.process(image);
        Log.i("YOLOW", "length image array: " + processed.getBuffer().asFloatBuffer().limit());

        FloatBuffer floatBuffer = processed.getBuffer().asFloatBuffer();

        StringJoiner stringJoiner = new StringJoiner(", ");

        for (int i = 0; i < floatBuffer.limit(); i++) {
            float val = floatBuffer.get(i);
            stringJoiner.add(String.valueOf(val));
        }

        Log.i("IMG", stringJoiner.toString());

        int n_col = (targets.classes.length + 4);
        int out_tensor_size = n_col * N_ANCHORS;

        FloatBuffer outBuffer = FloatBuffer.allocate(out_tensor_size);
        HashMap<Integer, Object> outHashmap = new HashMap<>();
        outHashmap.put(0, outBuffer);
        Log.i("YOLOW", "starting inference");

        NDArray<Float, D2> txt_feats = ConstructorsKt.ndarray(Multik.INSTANCE, targets.tensor.getFloatArray(), targets.classes.length, 512);

//        Log.i("YOLOW", "inputs: " + model.getInputTensorCount());
        Log.i("YOLOW", "signature keys: " + Arrays.toString(model.getSignatureKeys()));
        Log.i("YOLOW", "target tensor length: " + targets.tensor.getFloatArray().length);
        model.resizeInput(1, new int[]{1, targets.classes.length, EMB_DIM});
        model.allocateTensors();
        Tensor inp1 = model.getInputTensor(0);
        Tensor inp2 = model.getInputTensor(1);
        Log.i("YOLOW", "inp1: " + inp1.numElements() + "  inp2: " + inp2.numElements());

        Object[] inputs = new Object[]{processed.getBuffer().asFloatBuffer(), targets.tensor.getBuffer().asFloatBuffer()};
        model.runForMultipleInputsOutputs(inputs, outHashmap);

        TensorBuffer outTensor = TensorBuffer.createFixedSize(new int[]{1, N_ANCHORS, n_col}, DataType.FLOAT32);
        float[] outArray = outBuffer.array();
        outTensor.loadArray(outArray);

        assert outArray[1000] == outTensor.getFloatValue(1000);

        Log.i("YOLOW", "inferred: " + Arrays.toString(outBuffer.array()));

//        ArrayList<Integer> objects = YOLOW.classes_from_tensor(outBuffer, 0.5);
//        Log.i("YOLOW", "number of detected objects: " + objects.size());
//
//        HashSet<String> object_names = new HashSet<>();
//        for (int obj_idx: objects) {
//            object_names.add(targets.classes[obj_idx]);
//        }
//
//        Log.i("YOLOW", "detected classes: " + object_names);

        ArrayList<DetectUtils.Detect> detects = DetectUtils.non_maximum_suppression(outArray,  N_ANCHORS, IoUThresh, min_prob);

        for (DetectUtils.Detect detect : detects) {
            detect.label = targets.classes[detect.class_idx];
        }

        Log.i("YOLOW", "nb detections: " + detects.size());
        Log.i("YOLOW", "detections: " + detects);

        return detects;
    }
}
