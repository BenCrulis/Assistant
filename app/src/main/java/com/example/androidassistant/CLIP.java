package com.example.androidassistant;

import static com.example.androidassistant.utils.Image.getImagenetNormalizeOp;
import static com.example.androidassistant.utils.Image.getIntNormalizeOp;

import android.app.Application;
import android.util.Log;

import com.example.androidassistant.utils.ArrayUtils;
import com.example.androidassistant.utils.DetectUtils;
import com.example.androidassistant.utils.TFLite;

import org.jetbrains.kotlinx.multik.api.ConstructorsKt;
import org.jetbrains.kotlinx.multik.api.Engine;
import org.jetbrains.kotlinx.multik.api.EngineKt;
import org.jetbrains.kotlinx.multik.api.KEEngineType;
import org.jetbrains.kotlinx.multik.api.Multik;
import org.jetbrains.kotlinx.multik.api.math.Math;
import org.jetbrains.kotlinx.multik.ndarray.data.D1;
import org.jetbrains.kotlinx.multik.ndarray.data.D2;
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.StringJoiner;

public class CLIP {

    public static int EMB_DIM = 512;

    public static final Engine engine = EngineKt.enginesProvider().get(KEEngineType.INSTANCE);
    public static final Math math = YOLOW.engine.getMath();
    public static final ImageProcessor clipProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
//                    .add(new ResizeWithCropOrPadOp(672, 672))
//                    .add(getIntNormalizeOp())
                    .add(getImagenetNormalizeOp())
//                    .add(new TransposeOp())
                    .build();

    private ModelManager modelManager;

    public CLIP(ModelManager modelManager) {
        this.modelManager = modelManager;
    }

    private void log(String string) {
        Log.i("CLIP", string);
    }

    public TensorBuffer infer(TensorImage image) {
        AssistantApp assistantApp = AssistantApp.getInstance();
        Interpreter interpreter;
        try {
            interpreter = modelManager.getModel(assistantApp, "clip").get();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        TensorImage processed = clipProcessor.process(image);
        log("length image array: " + processed.getBuffer().asFloatBuffer().limit());

        FloatBuffer floatBuffer = processed.getBuffer().asFloatBuffer();

        StringJoiner stringJoiner = new StringJoiner(", ");

        for (int i = 0; i < floatBuffer.limit(); i++) {
            float val = floatBuffer.get(i);
            stringJoiner.add(String.valueOf(val));
        }

        FloatBuffer outBuffer = FloatBuffer.allocate(EMB_DIM);
        HashMap<Integer, Object> outHashmap = new HashMap<>();
        outHashmap.put(0, outBuffer);
        log("starting inference");

        log("signature keys: " + Arrays.toString(interpreter.getSignatureKeys()));
        interpreter.allocateTensors();
        Tensor inp1 = interpreter.getInputTensor(0);
        log("inp1: " + inp1.numElements());

        Object[] inputs = new Object[]{processed.getBuffer().asFloatBuffer()};
        interpreter.run(processed.getBuffer().asFloatBuffer(), outBuffer);

        TensorBuffer outTensor = TensorBuffer.createFixedSize(new int[]{EMB_DIM}, DataType.FLOAT32);
        float[] outArray = outBuffer.array();
        outTensor.loadArray(outArray);

        log("output array shape: " + Arrays.toString(outTensor.getShape()));

        return outTensor;

    }
}
