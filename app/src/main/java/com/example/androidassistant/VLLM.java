package com.example.androidassistant;

import android.app.Application;
import android.util.Log;

import com.example.androidassistant.utils.TFLite;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class VLLM {

    public static class LLMOutput {
        public final ByteBuffer new_state;
        public final ByteBuffer logits;

        public LLMOutput(ByteBuffer newState, ByteBuffer logits) {
            new_state = newState;
            this.logits = logits;
        }
    }

    public static TFLite.ModelWithBuffer loadMoondream(Application app, Interpreter.Options options) {
        String filename = "moondream-q2-tf.tflite";
        try {
            InputStream inputStream = app.getAssets().open(filename);
            int size = inputStream.available();
            TFLite.ModelWithBuffer modelWithBuffer = TFLite.loadModelWithSizeUnsafe(inputStream, size, options);
            inputStream.close();
            return modelWithBuffer;
        }
        catch (IOException e) {
            Log.e("moondream", "exception", e);
            System.exit(1);
        }
        return null;
    }

    public static int get_cache_size(int seq_len) {
        return 24 * 2 * 32 * 64 * seq_len * 4;
    }

    public static LLMOutput infer_from_embeddings(Interpreter model, ByteBuffer cache, ByteBuffer input_embs) {
        int SEQ_LEN = 1;
        Map<Integer, Object> output = new HashMap<>();
        output.put(1, ByteBuffer.allocate(SEQ_LEN * 51200 * 4));

        int cur_len = cache.limit() / (24*2*32*64*4);
        output.put(0, ByteBuffer.allocate(24 * 2 * 32 * 64 * (cur_len + 1) * 8));

        input_embs.rewind();

        model.resizeInput(1, new int[]{24, 2, 1, 32, cur_len, 64}, true);

        model.runForMultipleInputsOutputs(new Object[]{input_embs, cache}, output);

        ByteBuffer next_state = (ByteBuffer) output.get(0);
        ByteBuffer prob = (ByteBuffer) output.get(1);
        return new LLMOutput(next_state, prob);
    }

    public static void inference_loop(Interpreter model, ByteBuffer initial_cache, ByteBuffer initial_input_embs) {
        int GENERATED = 5;

        ByteBuffer cache = initial_cache;
        ByteBuffer input_embs = initial_input_embs;

        Log.i("moondream", "inference loop start");

        long timeBefore = System.currentTimeMillis();
        for (int i = 0; i < GENERATED; i++) {
            LLMOutput output = infer_from_embeddings(model, cache, input_embs);
            cache = output.new_state;
        }
        long elapsed = System.currentTimeMillis() - timeBefore;
        double elapsed_s = elapsed / 1000.0;
        double token_per_s = GENERATED / elapsed_s;
        Log.i("moondream", "inference loop done in " + elapsed_s + "s (" + token_per_s + " t/s).");
    }


    public static void testMoondream(Application app) {
        Interpreter.Options options = new Interpreter.Options()
                .setUseNNAPI(false)
                .setUseXNNPACK(true)
                .setCancellable(true)
                .setNumThreads(1);

        TFLite.ModelWithBuffer modelWithBuffer = loadMoondream(app, options);
        Interpreter model = modelWithBuffer.interpreter;

        String EMPTY_CACHE = "empty_cache";
        String COMPUTE_EMBEDDINGS = "compute_embeddings";

        String[] keys = model.getSignatureKeys();
        for (String key : keys) {
            Log.i("moondream", "key: " + key);
        }

        String[] empty_cache_inputs = model.getSignatureInputs(EMPTY_CACHE);
        for (String inp: empty_cache_inputs) {
            Log.i("moondream", "empty_cache input: " + inp);
        }
        String[] empty_cache_outputs = model.getSignatureOutputs(EMPTY_CACHE);
        Log.i("moondream", "outputs: " + Arrays.toString(empty_cache_outputs));
        Tensor cache_tensor = model.getOutputTensorFromSignature(empty_cache_outputs[0], EMPTY_CACHE);
        Log.i("moondream", "cache tensor shape: " + Arrays.toString(cache_tensor.shape()));

        HashMap<String, Object> initial_cache = new HashMap<>();
        initial_cache.put(empty_cache_outputs[0], cache_tensor);

        HashMap<String, Object> cache_input = new HashMap<>();
        cache_input.put("", null);

//        model.runSignature(cache_input, initial_cache, "empty_cache");

        String[] comp_emb_inputs = model.getSignatureInputs(COMPUTE_EMBEDDINGS);
        String[] comp_emb_outputs = model.getSignatureOutputs(COMPUTE_EMBEDDINGS);

        HashMap<String, Object> emb_input = new HashMap<>();

//        emb_input.put(comp_emb_inputs[0], TensorBuffer.createFixedSize(new int[] {1, 5}, DataType.INT32));

        int SEQ_LEN = 1;
        emb_input.put(comp_emb_inputs[0], ByteBuffer.allocate(SEQ_LEN*8));

        HashMap<String, Object> emb_output = new HashMap<>();
        emb_output.put(comp_emb_outputs[0], ByteBuffer.allocate(SEQ_LEN*2048*4));

//        model.resizeInput(2, new int[] {1, 5}); // cannot resize input of methods other than call

        model.runSignature(emb_input, emb_output, COMPUTE_EMBEDDINGS);

        for (String k : initial_cache.keySet()) {
            Log.i("moondream", "got " + k);
        }

        ByteBuffer token_emb = ByteBuffer.allocate(1*4);

        Map<Integer, Object> output = new HashMap<>();
        output.put(1, ByteBuffer.allocate(SEQ_LEN * 51200 * 4));
        output.put(0, ByteBuffer.allocate(24 * 2 * 32 * 64 * SEQ_LEN * 8));

        model.resizeInput(1, new int[]{24, 2, 1, 32, 0, 64}, true);
        //model.resizeInput(1, new int[]{0});
        //model.allocateTensors();

        for (int i=0; i < 2; i++) {
            Log.i("moondream", "output shape of input tensor " + i + " : " + Arrays.toString(model.getInputTensor(i).shape()));
            Log.i("moondream", "type of tensor " + i + " : " + model.getOutputTensor(i).dataType().toString());
        }

        for (int i=0; i < 2; i++) {
            Log.i("moondream", "output shape of output tensor " + i + " : " + Arrays.toString(model.getOutputTensor(i).shape()));
            Log.i("moondream", "type of tensor " + i + " : " + model.getOutputTensor(i).dataType().toString());
        }

        model.runForMultipleInputsOutputs(new Object[]{emb_output.get("output_0")}, output);

        ByteBuffer next_state = (ByteBuffer) output.get(0);
        ByteBuffer prob = (ByteBuffer) output.get(1);

        inference_loop(model, next_state.slice(), (ByteBuffer) emb_output.get("output_0"));

        // clean up memory
        Log.i("moondream", "cleaning up memory");
        modelWithBuffer.unsafeDeallocate();
        model.close();
    }

}
