package com.example.androidassistant;

import android.app.Application;
import android.util.Log;

import com.example.androidassistant.utils.TFLite;
import com.example.androidassistant.utils.alloc.NativeByteBuffer;
import com.example.androidassistant.utils.tokenization.MoondreamTokenizer;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class VLLM {

    public static String TAG = "VLLM";

    public static String COMPUTE_EMBEDDINGS_KEY = "compute_embeddings";
    public static String CALL_KEY = "call";

    public static int N_OUTPUTS = 51200;

    public static String describe_prompt = "\n\nQuestion: Describe this image.\n\nAnswer:";

    public static int MAX_GENERATED_TOKENS = 200;

    public static final ImageProcessor visionProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(378, 378, ResizeOp.ResizeMethod.BILINEAR))
//                    .add(getImagenetNormalizeOp())
//                    .add(new TransposeOp())
//                    .add(getIntNormalizeOp())
                    .add(new NormalizeOp(new float[]{127.5f, 127.5f, 127.5f}, new float[]{127.5f, 127.5f, 127.5f}))
                    .build();

    public static class LLMOutput {
        public final ByteBuffer new_state;
        public final FloatBuffer logits;

        public LLMOutput(ByteBuffer newState, FloatBuffer logits) {
            new_state = newState;
            this.logits = logits;
        }
    }

    public static MoondreamTokenizer loadTokenizer(AssistantApp app) {
        String vocab_path = app.getAssetFilePath("vocab.json");
        String merges_path = app.getAssetFilePath("merges.txt");

        MoondreamTokenizer tokenizer = MoondreamTokenizer.from_files(vocab_path, merges_path);
        return tokenizer;
    }

    public static TFLite.ModelWithBuffer loadMoondream(Application app, Interpreter.Options options) {
        String filename = "moondream-q2-matmul.tflite";
//        String filename = "moondream-q2-tf.tflite";
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

    public static LLMOutput infer_from_embeddings(Interpreter model, ByteBuffer cache, FloatBuffer input_embs) {
        int new_tokens = input_embs.capacity()/2048;
        Map<Integer, Object> output = new HashMap<>();
//        output.put(1, ByteBuffer.allocate(SEQ_LEN * 51200 * 4));

        TensorBuffer output_logits_tensorbuffer = TensorBuffer.createFixedSize(new int[]{new_tokens, N_OUTPUTS}, DataType.FLOAT32);
        output.put(1, output_logits_tensorbuffer.getBuffer());

        int cur_len = 0;
        if (cache != null) {
            cur_len = cache.limit() / (24*2*32*64*4);
            cache.rewind();
        }
//        output.put(0, ByteBuffer.allocate(24 * 2 * 32 * 64 * (cur_len + 1) * 8));

        int max_size = 780;

        if (cache == null) {
            //output_cache_tensorbuffer = TensorBuffer.createFixedSize(new int[]{max_size, 24, 2, 1, 32, 64}, DataType.FLOAT32);
            // output.put(0, output_cache_tensorbuffer.getBuffer());
            ByteBuffer output_cache_tensorbuffer = NativeByteBuffer.createBuffer(max_size * 24 * 2 * 32 * 64 * 4);
            output_cache_tensorbuffer.order(ByteOrder.LITTLE_ENDIAN); // /!\ very important
            output.put(0, output_cache_tensorbuffer);
        }
        else {
            TensorBuffer output_cache_tensorbuffer = TensorBuffer.createFixedSize(new int[]{new_tokens, 24, 2, 1, 32, 64}, DataType.FLOAT32);
            output.put(0, output_cache_tensorbuffer.getBuffer());
        }

        input_embs.rewind();

        model.resizeInput(0, new int[]{1, new_tokens, 2048}, true);

        if (cache == null) {
            model.resizeInput(1, new int[]{cur_len, 24, 2, 1, 32, 64}, true);
            model.runForMultipleInputsOutputs(new Object[]{input_embs}, output);
        }
        else {
            model.resizeInput(1, new int[]{cur_len, 24, 2, 1, 32, 64}, true);
            model.runForMultipleInputsOutputs(new Object[]{input_embs, cache.slice()}, output);
        }

//        ByteBuffer next_state = (ByteBuffer) output.get(0);
//        ByteBuffer prob = (ByteBuffer) output.get(1);

        ByteBuffer next_state = (ByteBuffer) output.get(0);
        next_state.limit(next_state.position());
        output_logits_tensorbuffer.getBuffer().rewind();
        FloatBuffer prob = output_logits_tensorbuffer.getBuffer().asFloatBuffer();
        return new LLMOutput(next_state, prob);
    }

    public static void inference_loop(Interpreter model, ByteBuffer initial_cache, ByteBuffer initial_input_embs) {
        int GENERATED = 1;

        ByteBuffer cache = initial_cache;

        if (cache == null) {
//            cache = ByteBuffer.allocate(0);
        }

        ByteBuffer input_embs = initial_input_embs;

        Log.i("moondream", "inference loop start");

        long timeBefore = System.currentTimeMillis();
        for (int i = 0; i < GENERATED; i++) {

            int new_tokens = input_embs.asFloatBuffer().capacity()/2048;

            LLMOutput output = infer_from_embeddings(model, cache, input_embs.asFloatBuffer());
            //cache = output.new_state;

//            cache = ByteBuffer.allocate(cache.capacity() + output.new_state.capacity());
//            cache.position(cache.capacity());
//            cache.put(output.new_state);
//            cache.rewind();

            output.logits.rewind();

            int N_DISPLAY = 15;

            Log.i(TAG, "inference_loop: iteration " + i);
            FloatBuffer out = output.logits;
            out.rewind();

            for (int j = 0; j < new_tokens; j++) {
                float[] output_float = new float[N_DISPLAY];
                out.position(j * N_OUTPUTS);
                out.get(output_float, 0, N_DISPLAY);
                Log.i("moondream", "output " + i + ": " + Arrays.toString(output_float));
                out.position(j * N_OUTPUTS);
                out.limit((j+1) * N_OUTPUTS);
                int predicted_token = floatBufferArgmax(out.slice());
                out.limit(out.capacity());
                Log.i("moondream", "predicted token: " + predicted_token);
            }

        }
        long elapsed = System.currentTimeMillis() - timeBefore;
        double elapsed_s = elapsed / 1000.0;
        double token_per_s = GENERATED / elapsed_s;
        Log.i("moondream", "inference loop done in " + elapsed_s + "s (" + token_per_s + " t/s).");
    }

    public static FloatBuffer computeDescriptionPromptEmbedding(AssistantApp app, TensorImage tensorImage) throws IOException {
        ModelManager modelManager = app.getModelManager();

        Interpreter vision_encoder = modelManager.getModel(app, "moondream_vision_enc").get();

        // floatbuffers are naturally in LITTLE ENDIAN
        FloatBuffer out_embed_floatbuffer = FloatBuffer.allocate(729*2048);

        Log.i(TAG, "computeDescriptionPromptEmbedding: byteorder " + out_embed_floatbuffer.order());

        Log.i(TAG, "preprocessing image");

        tensorImage = visionProcessor.process(tensorImage);

        Log.i(TAG, "computing embedding");

        vision_encoder.run(tensorImage.getBuffer(), out_embed_floatbuffer);

        Log.i(TAG, "unloading vision encoder");

        modelManager.unloadModel("moondream_vision_enc");

        Log.i(TAG, "embedding: " + Arrays.toString(out_embed_floatbuffer.array()));
        return out_embed_floatbuffer;
    }

    public static FloatBuffer computeSingleTokenEmbedding(Interpreter moondream, int token_id) {
        assert token_id <= 50255;

        LongBuffer prompt_tokens_buffer = LongBuffer.allocate(1); // allocate one long
        prompt_tokens_buffer.rewind();
        prompt_tokens_buffer.put(token_id);
        prompt_tokens_buffer.rewind();

        // compute token embedding
        HashMap<String, Object> input_map = new HashMap<>();

        input_map.put("token_ids", prompt_tokens_buffer);
        HashMap<String, Object> output_map = new HashMap<>();

        FloatBuffer out = FloatBuffer.allocate(2048); // allocate a single vector of floats

        output_map.put("output_0", out);

        moondream.allocateTensors();
        moondream.runSignature(input_map, output_map, COMPUTE_EMBEDDINGS_KEY);

        return out;
    }

    public static ByteBuffer debug_cache(int len) {
        ByteBuffer out = ByteBuffer.allocateDirect(len * 24 * 2 * 32 * 64 * 4);
        out.order(ByteOrder.LITTLE_ENDIAN);
        out.rewind();

        FloatBuffer outFloat = out.asFloatBuffer();

        while (outFloat.hasRemaining()) {
            outFloat.put(1f);
        }
        out.rewind();
        return out;
    }

    public static FloatBuffer computeTokenEmbedding(Interpreter moondream, List<Integer> tokens) {
        FloatBuffer final_token_embeddings = FloatBuffer.allocate(tokens.size() * 2048);
        final_token_embeddings.rewind();

        Log.i(TAG, "computeTokenEmbedding: computing token embeddings");

        for (int token_id : tokens) {
            Log.i(TAG, "computeTokenEmbedding: token id: " + token_id);
            FloatBuffer out = computeSingleTokenEmbedding(moondream, token_id);
            Log.i(TAG, "computeTokenEmbedding: handled token: " + token_id);

            // append to final embedding buffer
            final_token_embeddings.put(out.array());
        }

        return final_token_embeddings;
    }

    public static int floatBufferArgmax(FloatBuffer logits) {
        int argmax = 0;
        float max = logits.get();
        int i = 0;
        while (logits.hasRemaining()) {
            i++;
            float val = logits.get();
            if (val > max) {
                max = val;
                argmax = i;
            }
        }
        return argmax;
    }

    public static String computeImageDescription(AssistantApp app, FloatBuffer image_embedding) throws IOException {
        ModelManager modelManager = app.getModelManager();
        modelManager.unloadAll(); // make sure we have the maximum amount of memory available
        Log.i(TAG, "loading Moondream");

        Interpreter.Options options = new Interpreter.Options()
                .setNumThreads(4)
                .setUseXNNPACK(false)
                .setCancellable(true);

        Interpreter moondream = modelManager.getModel(app, "moondream", options).get();

        MoondreamTokenizer tokenizer = app.getTokenizer();

        ArrayList<Integer> prompt_tokens = tokenizer.encode(VLLM.describe_prompt);

        // compute token embeddings
        Log.i(TAG, "computeImageDescription: first token id: " + prompt_tokens.get(0));

        FloatBuffer text_embedding = computeTokenEmbedding(moondream, prompt_tokens);

        Log.i(TAG, "token embeddings: " + Arrays.toString(text_embedding.array()));

        FloatBuffer prompt_embedding = FloatBuffer.allocate(image_embedding.capacity() + text_embedding.capacity());

        image_embedding.rewind();
        text_embedding.rewind();
        prompt_embedding.put(image_embedding);
        prompt_embedding.put(text_embedding);
        prompt_embedding.rewind();

        image_embedding = null;
        text_embedding = null;

        System.gc();

        ByteBuffer cache = null;

        LLMOutput output = null;

        int n_generated = 0;

        int token_id = -1;

        ArrayList<Integer> generated_tokens = new ArrayList<>();

        int remaining_cache_size = 0;

        try {

            do {

                output = infer_from_embeddings(moondream, cache, prompt_embedding);
//            cache = output.new_state;

                if (cache == null) {
                    cache = output.new_state;
                } else {
                    cache.limit(cache.limit() + output.new_state.capacity());
                    cache.position(cache.limit() - output.new_state.capacity());
                    cache.put(output.new_state);
                    cache.rewind();
                }

                output.logits.rewind();
                output.logits.position(output.logits.capacity() - N_OUTPUTS);

                token_id = floatBufferArgmax(output.logits);
                output = null;
                Log.i(TAG, "computeImageDescription: generated token: " + token_id);
                prompt_embedding = computeSingleTokenEmbedding(moondream, token_id);

                generated_tokens.add(token_id);

                remaining_cache_size = (cache.capacity() - cache.limit()) / (24 * 2 * 32 * 64 * 4);

                n_generated++;
            }
            while (n_generated < 15 && token_id != tokenizer.get_eos_token() && remaining_cache_size > 0);

        }
        catch (Exception e) {
            throw e;
        }
        finally {
            if (cache != null) { // deallocating natively allocated ByteBuffer
                NativeByteBuffer.deallocateBuffer(cache);
            }
        }

        String decoded = tokenizer.decode(generated_tokens);

        return decoded;

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
        //initial_cache.put(empty_cache_outputs[0], cache_tensor);
        initial_cache.put(empty_cache_outputs[0], debug_cache(10));

        HashMap<String, Object> cache_input = new HashMap<>();
        cache_input.put("", null);

//        model.runSignature(cache_input, initial_cache, "empty_cache");

        String[] comp_emb_inputs = model.getSignatureInputs(COMPUTE_EMBEDDINGS);
        String[] comp_emb_outputs = model.getSignatureOutputs(COMPUTE_EMBEDDINGS);

        HashMap<String, Object> emb_input = new HashMap<>();

//        emb_input.put(comp_emb_inputs[0], TensorBuffer.createFixedSize(new int[] {1, 5}, DataType.INT32));

        int SEQ_LEN = 1;
        ByteBuffer inp_token_bb = ByteBuffer.allocate(SEQ_LEN*8);
        // put zeros
        inp_token_bb.putLong(0);
        inp_token_bb.rewind();

        emb_input.put(comp_emb_inputs[0], inp_token_bb);

        HashMap<String, Object> emb_output = new HashMap<>();
//        ByteBuffer out_emb_bb = ByteBuffer.allocate(SEQ_LEN*2048*4);
        TensorBuffer out_emb_bb = TensorBuffer.createFixedSize(new int[]{SEQ_LEN, 2048}, DataType.FLOAT32);
        emb_output.put(comp_emb_outputs[0], out_emb_bb.getBuffer());

        ArrayList<Integer> zero_tokens = new ArrayList<>();
        for (int i = 0; i < 11; i++) {
            zero_tokens.add(0);
        }

        FloatBuffer zero_emb = computeTokenEmbedding(model, zero_tokens);

        //emb_output.put(comp_emb_outputs[0], zero_emb);

        model.runSignature(emb_input, emb_output, COMPUTE_EMBEDDINGS);

        // log input token as an in64
        inp_token_bb.rewind();
        Log.i("moondream", "input token: " + inp_token_bb.getLong());
        inp_token_bb.rewind();

        // log output embedding (5 first terms)
        float[] out_emb = Arrays.copyOfRange(out_emb_bb.getFloatArray(), 0, 15);
        Log.i("moondream", "output embedding: " + Arrays.toString(out_emb));

        for (String k : initial_cache.keySet()) {
            Log.i("moondream", "got " + k);
        }

        Map<Integer, Object> output = new HashMap<>();
        output.put(1, ByteBuffer.allocate(SEQ_LEN * N_OUTPUTS * 4));
        output.put(0, ByteBuffer.allocate(24 * 2 * 32 * 64 * SEQ_LEN * 4));

        model.resizeInput(1, new int[]{0, 24, 2, 1, 32, 64}, true);
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
//
//        model.runForMultipleInputsOutputs(new Object[]{emb_output.get("output_0")}, output);
//
//        ByteBuffer next_state = (ByteBuffer) output.get(0);
//        ByteBuffer prob = (ByteBuffer) output.get(1);

//        inference_loop(model, next_state.slice(), (ByteBuffer) emb_output.get("output_0"));
        //inference_loop(model,  debug_cache(10), (ByteBuffer) emb_output.get("output_0"));

        ByteBuffer zero_emb_bytebuffer = ByteBuffer.allocate(zero_emb.capacity() * 4);
        zero_emb.rewind();
        zero_emb_bytebuffer.rewind();
        zero_emb_bytebuffer.asFloatBuffer().put(zero_emb);

        inference_loop(model,  debug_cache(10), zero_emb_bytebuffer);
        //inference_loop(model,  null, zero_emb_bytebuffer);

        // clean up memory
        Log.i("moondream", "cleaning up memory");
        modelWithBuffer.unsafeDeallocate();
        model.close();
    }


    public static void testTokenizer(AssistantApp app) {
        String vocab_path = app.getAssetFilePath("vocab.json");
        String merges_path = app.getAssetFilePath("merges.txt");

        MoondreamTokenizer tokenizer = MoondreamTokenizer.from_files(vocab_path, merges_path);

        String to_encode = "<|endoftext|>Hello world!\nMy name is John Doe.<|endoftext|>";
        to_encode = VLLM.describe_prompt;

        Log.i("moondream", "encoding: " + to_encode);

        List<Integer> ids = tokenizer.encode(to_encode);

        Log.i("moondream", "tokenized: " + Arrays.toString(ids.toArray()));

        String decoded = tokenizer.decode(ids);

        Log.i("moondream", "decoded: " + decoded);

        assert to_encode.equals(decoded);
    }


}
