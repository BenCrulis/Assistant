package com.example.androidassistant;

import android.app.Activity;
import android.util.Log;

import com.example.androidassistant.activities.ObjectSavingActivity;
import com.example.androidassistant.database.Embedding;
import com.example.androidassistant.utils.TensorUtils;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.util.Arrays;

public class DirtinessDetection {
    private static String TAG = "DirtinessDetect";

    private static double LOGIT_SCALE = 100.0;

    public static float computeCleanlinessScore(TensorBufferFloat embedding,
                                     TensorBufferFloat cleanEmb,
                                     TensorBufferFloat dirtyEmb) {

        double dotClean = TensorUtils.dot(embedding, cleanEmb);
        double dotDirty = TensorUtils.dot(embedding, dirtyEmb);

        TensorBufferFloat values = TensorUtils.FloatTensorFromFloatValues(
                (float) dotClean,
                (float) dotDirty);

        values = TensorUtils.map(values, x -> (float) (x * LOGIT_SCALE));

        Log.i(TAG, "computeCleanlinessScore: value 1: " + values.getFloatValue(0));

        Log.i(TAG, "computeCleanlinessScore: logits: " + Arrays.toString(values.getFloatArray()));

        TensorBufferFloat scores = TensorUtils.softmax(values);

        float[] arr = scores.getFloatArray();

        Log.i(TAG, "computeCleanlinessScore: scores: " + Arrays.toString(arr));

        return arr[0];
    }


    public static void detectDirtiness(AssistantApp app, Activity activity) {
        Log.i(TAG, "detectDirtiness: detecting dirtiness");
        AssistantApp assistantApp = AssistantApp.getInstance();

        TensorBufferFloat cleanEmb = TensorUtils.readVectorFromFile(app.data_base_path + "/clean_emb.txt");
//        TensorBufferFloat dirtyEmb = TensorUtils.readVectorFromFile(app.data_base_path + "/dirty_emb.txt");
        TensorBufferFloat dirtyEmb = TensorUtils.readVectorFromFile(app.data_base_path + "/unclean_emb.txt");

        // normalize embeddings
        cleanEmb = TensorUtils.normalize(cleanEmb);
        dirtyEmb = TensorUtils.normalize(dirtyEmb);

        final TensorBufferFloat cleanEmbFinal = cleanEmb;
        final TensorBufferFloat dirtyEmbFinal = dirtyEmb;

        assistantApp.takePhoto(activity, tensorImage -> {
            Log.i(TAG, "detectDirtiness: successfully got tensorImage object");

            CLIP clip = new CLIP(assistantApp.getModelManager());

            TensorBuffer emb = clip.infer(tensorImage);

            Log.i("detectDirtiness", "shape: " + Arrays.toString(emb.getShape()));
            Log.i("detectDirtiness", "emb[:10]: " +
                    Arrays.toString(Arrays.copyOfRange(emb.getFloatArray(), 0, 10)));

            float[] floatEmb = emb.getFloatArray();
            assert floatEmb.length == 512;

            float cleanScore = computeCleanlinessScore(TensorUtils.normalize((TensorBufferFloat) emb),
                    cleanEmbFinal, dirtyEmbFinal);

            float roundedScore = Math.round(cleanScore * 100.0f);

            String toSpeak = "" + roundedScore + "% propre.";

            AssistantApp.getInstance().queueSpeak(toSpeak);
        });
    }
}
