package com.example.androidassistant;

import static com.example.androidassistant.utils.Image.getImagenetNormalizeOp;

import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;

import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.lifecycle.ProcessLifecycleOwner;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;

public class AssistantApp extends Application {
    private AssistantApp singleton = null;
    TextToSpeech tts = null;
    CountDownLatch ttsCountDownLatch = null;

    String[] imagenet_labels = null;
    ImageCapture imageCapture = null;
    private static final int MY_CAMERA_REQUEST_CODE = 100; // Any unique integer
    static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int SPEECH_REQUEST_CODE = 0;

    private final ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .add(getImagenetNormalizeOp())
                    .build();


    private final ImageProcessor yoloProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                    .add(getImagenetNormalizeOp())
                    .build();

    private final ImageProcessor imageProcessorDepth =
            new ImageProcessor.Builder()
                    //.add(new Rot90Op(0))
                    .add(new ResizeOp(518, 518, ResizeOp.ResizeMethod.BILINEAR))
                    .add(getImagenetNormalizeOp())
                    .build();

    private Interpreter interpreter = null;
    private Interpreter depthAnythingModel = null;
    private Interpreter yoloModel = null;

    public AssistantApp getInstance(){
        return singleton;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        init();
        singleton = this;
    }

    public void init() {
        Log.i("INIT", "initializing everything");

        // setup TTS
        ttsCountDownLatch = new CountDownLatch(1);
        setupTTS();
        long threadId = Thread.currentThread().getId();
        Log.i("TTS", "current thread id: " + threadId);
        Log.i("TTS", "waiting for semaphore");
//        try {
//            ttsCountDownLatch.await();
//        } catch (InterruptedException e) {
//            throw new RuntimeException(e);
//        }
        Log.i("TTS", "continuing after semaphore");


//        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
//            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
//        }

        WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);

        imageCapture = new ImageCapture.Builder()
//                .setTargetRotation(windowManager.getDefaultDisplay().getRotation()) // same as putting nothing
//                .setTargetRotation(Surface.ROTATION_0)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
                .build();
        ProcessCameraProvider cameraProvider = null;
        try {
            cameraProvider = ProcessCameraProvider.getInstance(this).get();
        } catch (Exception e) {
            Log.e("PROVIDER", e.toString());
            queueSpeak("Erreur au moment de récupérer la caméra");
        }

        if (cameraProvider == null) {
            System.exit(1);
        }

        cameraProvider.bindToLifecycle(ProcessLifecycleOwner.get(), CameraSelector.DEFAULT_BACK_CAMERA, imageCapture);

        //this.displaySpeechRecognizer();

        // load label file

        try (InputStream inputStream = getAssets().open("imagenet_labels.txt")) {
            BufferedReader bufferedReader = new BufferedReader(
                    new InputStreamReader(inputStream, StandardCharsets.UTF_8));
            imagenet_labels = bufferedReader.lines().toArray(String[]::new);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        // TFLite part
        Interpreter.Options options = new Interpreter.Options()
                .setUseNNAPI(false)
                .setUseXNNPACK(false)
                .setNumThreads(1);

        String basePath = getDataDir().getAbsolutePath() + "/";

        String modelName = "quicknet.tflite"; // input/output in float
//        String modelName = "mobilenetv1.tflite"; // input/output in byte

//        File modelFile = new File(basePath + modelName);
//        Log.i("info", "opening asset model " + modelName);
//        try (InputStream assetStream = getAssets().open(modelName)) {
//            Log.i("info", "copying asset model");
//            Files.copy(assetStream, Paths.get(basePath + modelName), StandardCopyOption.REPLACE_EXISTING);
//
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
        File modelFile = new File(basePath + modelName);
        copyAssetToDisk(basePath + modelName);

        copyAssetToDisk(basePath + "depth_anything_small.tflite");
        loadAnythingDepth(options);

        copyAssetToDisk(basePath + "yolov8m_int8.tflite");
        loadYOLO(options);

        copyAssetToDisk(basePath + "Great_White_Shark.jpg");
        copyAssetToDisk(basePath + "test_image.png");

        // testing inference
        Log.i("info", "loading model");
        try {
            interpreter = new Interpreter(modelFile, options);
            int count = interpreter.getOutputTensorCount();
            Log.i("info", "number of output tensors: " + count);
            Tensor inputTensor = interpreter.getInputTensor(0);
            Log.i("info", "data type: " + inputTensor.dataType().toString());
            Log.i("info", "num dimensions: " + inputTensor.numDimensions());
            Log.i("info", "shape: " + Arrays.toString(inputTensor.shape()));
            //float[][][][] inputs = new float[1][224][224][3];

            // Image tensors are C-style row-major
//            Bitmap diskImage = BitmapFactory.decodeFile(basePath + "Great_White_Shark.jpg");
            Bitmap diskImage = BitmapFactory.decodeFile(basePath + "test_image.png");
            TensorImage tensorImage = TensorImage.fromBitmap(diskImage);
            Log.i("GWS", "raw: " + Arrays.toString(tensorImage.getTensorBuffer().getFloatArray()));
            tensorImage = imageProcessor.process(tensorImage);

            Log.i("info", "before GWS");
            //Log.i("GWS", Arrays.toString(tensorImage.getBuffer().asFloatBuffer().array()));
            Log.i("GWS", Arrays.toString(tensorImage.getTensorBuffer().getFloatArray()));

            FloatBuffer inputs = FloatBuffer.allocate(224*224*3);
            Arrays.fill(inputs.array(), 1.0f);
            inputs.rewind();
            FloatBuffer outputs = FloatBuffer.allocate(1000);
//            interpreter.resizeInput(0, new int[]{3*224*224});
            interpreter.run(tensorImage.getBuffer().asFloatBuffer(), outputs);
            Log.i("info", "after inference");

            Log.i("info", Arrays.toString(outputs.array()));
        }
        catch (Exception e) {
            Log.e("MODEL", e.toString());
        }
    }



    private void copyAssetToDisk(String path) {
        File modelFile = new File(path);
        Log.i("info", "opening asset model " + modelFile.getName());
        try (InputStream assetStream = getAssets().open(modelFile.getName())) {
            Log.i("info", "copying asset model");
            Files.copy(assetStream, Paths.get(path), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void loadAnythingDepth(Interpreter.Options options) {
        try {
            // getAssets().open("depth_anything_small.tflite");
            String basePath = getDataDir().getAbsolutePath() + "/";
            String modelName = "depth_anything_small.tflite";
            File modelFile = new File(basePath + modelName);
            if (!modelFile.exists()) {
                Log.e("DEPTH", "model file does not exist at " + modelFile.getPath());
            }
            depthAnythingModel = new Interpreter(modelFile, options);
        }
        catch (Exception e) {
            Log.e("DepthAnything", e.toString());
            System.exit(1);
        }
    }

    private void loadYOLO(Interpreter.Options options) {
        String filename = "yolov8m_int8.tflite";
        int k = 0;
        try {
            byte[] buffer = new byte[65536];
            InputStream inputStream = getAssets().open(filename);
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            k = inputStream.read(buffer);
            while (k > 0) {
                outputStream.write(buffer, 0, k);
                k = inputStream.read(buffer);
            }
            Log.i("bytebuffer", "k: " + k);
            outputStream.flush();
            byte[] modelBytes = outputStream.toByteArray();
            Log.i("bytebuffer", "byte[] array size: " + modelBytes.length);
            ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
            Log.i("bytebuffer", "bytebuffer size: " + modelBuffer.capacity());
            modelBuffer.put(modelBytes);
            modelBuffer.rewind();
            Log.i("bytebuffer", "is direct: " + modelBuffer.isDirect());
            yoloModel = new Interpreter(modelBuffer, options);
            inputStream.close();
        }
        catch (Exception e) {
            Log.e("YOLO", "exception", e);
            Log.e("YOLO", "k was " + k);
            System.exit(1);
        }
    }


    public void setupTTS() {
        Log.i("TTS", "TTS setup started");
        double timeBefore = System.currentTimeMillis();
        tts = new TextToSpeech(this, i -> {
            double elapsed = System.currentTimeMillis() - timeBefore;
            Log.i("TTS", "TTS ready (" + (elapsed / 1000) + "s)");
            tts.setLanguage(Locale.FRENCH);
            long threadId = Thread.currentThread().getId();
            Log.i("TTS", "callback thread id: " + threadId);
        });
    }
    public void queueSpeak(String speak) {
        Log.i("TTS", "adding to queue: \"" + speak + "\"");
        tts.speak(speak, TextToSpeech.QUEUE_ADD, null, null);
    }

    public void flushSpeak(String speak) {
        Log.i("TTS", "flushing queue and saying: \"" + speak + "\"");
        tts.speak(speak, TextToSpeech.QUEUE_FLUSH, null, null);
    }

}
