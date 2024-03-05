package com.example.androidassistant;

import static com.example.androidassistant.utils.Image.getImagenetNormalizeOp;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.Surface;
import android.view.WindowManager;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.lifecycle.ProcessLifecycleOwner;

import com.example.androidassistant.utils.DetectUtils;
import com.example.androidassistant.utils.Image;

import org.tensorflow.lite.DataType;
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
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

public class AssistantApp extends Application {
    private static AssistantApp singleton;
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

    public static final int DEPTH_IMAGE_SIZE = 518;
    private final ImageProcessor imageProcessorDepth =
            new ImageProcessor.Builder()
                    //.add(new Rot90Op(0))
                    .add(new ResizeOp(DEPTH_IMAGE_SIZE, DEPTH_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    .add(getImagenetNormalizeOp())
                    .build();

    private Interpreter interpreter = null;
    private Interpreter depthAnythingModel = null;
    private Interpreter yoloModel = null;

    private YOLOW yolow = null;


    @Override
    public void onCreate() {
        super.onCreate();
        init();
        singleton = this;
    }

    public static AssistantApp getInstance() {
        return AssistantApp.singleton;
    }

    public void init() {
        Log.i("INIT", "initializing everything");

        Context context = getApplicationContext();
        ActivityManager result = (ActivityManager)context.getSystemService(Context.ACTIVITY_SERVICE);
        Log.i("INIT", "large memory class: " + result.getLargeMemoryClass());

        // doesn't work
        //ByteBuffer buffer = ByteBuffer.allocateDirect(1000*1000*1000);

        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        Log.i("INIT", "max memory: " + maxMemory / (1000*1000));

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
                .setTargetRotation(Surface.ROTATION_0)
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

//        copyAssetToDisk(basePath + "yolov8m_int8.tflite");
//        loadYOLO(options);

        yolow = new YOLOW("l");
        Log.i("YOLOW", "loading YOLO-World");
        yolow.loadYOLOW(this, options);
        Log.i("YOLOW", "loading YOLO-World targets");
        YOLOW.YOLOWTargets general_purpose_targets = YOLOW.load_yolow_classes_file(this, "general_purpose_classes.txt");
        yolow.add_targets(general_purpose_targets, "general");

        Log.i("YOLOW", "target classes: " + String.join(", ", general_purpose_targets.classes));
        Log.i("debug", "targets: " + Arrays.toString(general_purpose_targets.tensor.getFloatArray()));

        copyAssetToDisk(basePath + "Great_White_Shark.jpg");
        copyAssetToDisk(basePath + "test_image.png");
        copyAssetToDisk(basePath + "gmu_scene_2_rgb_462.png");

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


            // YOLOW test
            diskImage = BitmapFactory.decodeFile(basePath + "gmu_scene_2_rgb_462.png");
            tensorImage = TensorImage.fromBitmap(diskImage);
            Log.i("YOLOW", "inference test");
            yolow.infer(tensorImage, "general", 0.5, 0.6);


        }
        catch (Exception e) {
            Log.e("MODEL", e.toString());
        }
    }

    public void takePhoto(MainActivity activity, int rot) {
        imageCapture.takePicture(Executors.newSingleThreadExecutor(), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                boolean imageClassifier = false;
                boolean yolow_pred = true;
                boolean depth_pred = false;

                Log.i("PHOTO", "photo successfully taken");
//                        queueSpeak("J'ai pris une photo, j'analyse...");
                int height = image.getHeight();
                int width = image.getWidth();
                int planes = image.getPlanes().length;
                int imageRot = image.getImageInfo().getRotationDegrees();
//                int rotation_compensate_k = -Image.compute_k_from_degrees(imageRot);
                int rotation_compensate_k = rot;// + Image.compute_k_from_degrees(imageRot);
                //queueSpeak("k=" + rotation_compensate_k + ", rota=" + imageRot);
                Log.i("PHOTO", "width: " + width + ", height: " + height);
                Log.i("PHOTO", "rotation: " + imageRot + "°");
                if (height > width) {
                    //queueSpeak("Photo anormale, la hauteur est plus grande que la largeur.");
                }
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
                Log.i("buffer", "toString: " + Arrays.toString(intArray));

                // Create a TensorImage object. This creates the tensor of the corresponding
                // tensor type that the TensorFlow Lite interpreter needs.
                TensorImage tensorImage224 = new TensorImage(DataType.FLOAT32);
                // Analysis code for every frame
                // Preprocess the image
                tensorImage224.load(bitmap);
                long beforeClassify = System.currentTimeMillis();
                if (rotation_compensate_k != 0) {
                    Log.i("PHOTO", "rotating photo");
                    //queueSpeak("je tourne la photo");
//                            tensorImage224 = rotate90processor.process(tensorImage224);
                    tensorImage224 = Image.rotate90(tensorImage224, rotation_compensate_k);

                }
                tensorImage224 = imageProcessor.process(tensorImage224);
                long elapsedClassify = System.currentTimeMillis() - beforeClassify;
                Log.i("classify", "classify preprocessing: " + elapsedClassify + " ms");

                float[] im_array = tensorImage224.getTensorBuffer().getFloatArray();
                float[] top_left = Arrays.copyOfRange(im_array, 10*3, 11*3);
                float[] top_right = Arrays.copyOfRange(im_array, (224-11)*3, (224-10)*3);
                Log.i("COLOR", "top left: " + Arrays.toString(top_left) + " top right: " + Arrays.toString(top_right));

                if (imageClassifier) {

                    FloatBuffer output = FloatBuffer.allocate(1000);

                    interpreter.run(tensorImage224.getBuffer().asFloatBuffer(), output);

                    float[] outputArray = output.array();

                    int maxAt = 0;

                    for (int i = 0; i < outputArray.length; i++) {
                        maxAt = outputArray[i] > outputArray[maxAt] ? i : maxAt;
                    }

                    String label = imagenet_labels[maxAt];
                    float prob = outputArray[maxAt];
                    long percent = Math.round(prob * 100.0);

                    queueSpeak("Classe " + maxAt + ", " + label + " " + percent + " pourcents");
                }

                Bitmap finalBitmap = bitmap;
                activity.runOnUiThread(() -> {
                    activity.displayPhoto(finalBitmap, imageRot);
                });

                // yolo general prediction
                if (yolow_pred) {
                    TensorImage tensorImageYolow = new TensorImage(DataType.FLOAT32);
                    // Analysis code for every frame
                    // Preprocess the image
                    tensorImageYolow.load(bitmap);
                    if (rotation_compensate_k != 0) {
                        tensorImageYolow = Image.rotate90(tensorImageYolow, rotation_compensate_k);
                    }
                    ArrayList<DetectUtils.Detect> detected = yolow.infer(tensorImageYolow, "general", 0.5, 0.3);

                    if (detected.size() == 0) {
                        queueSpeak("Je ne vois aucun objet.");
                    } else {
                        StringBuilder detectionSb = new StringBuilder();
                        detectionSb.append("Je vois: ");
                        for (DetectUtils.Detect detect : detected) {
                            detectionSb.append(detect.label);
                            detectionSb.append(", ");
                        }
                        queueSpeak(detectionSb.toString());
                    }
                }

                // depth prediction
                if (depth_pred) {
                    TensorImage tensorImageDepth = new TensorImage(DataType.FLOAT32);
                    tensorImageDepth.load(bitmap);
                    long beforeDepth = System.currentTimeMillis();
                    if (rotation_compensate_k != 0) {
//                            tensorImageDepth = rotate90processor.process(tensorImageDepth);
                        tensorImageDepth = Image.rotate90(tensorImageDepth, rotation_compensate_k);
                    }
                    tensorImageDepth = imageProcessorDepth.process(tensorImageDepth);
                    long elapsedDepth = System.currentTimeMillis() - beforeDepth;
                    Log.i("depth", "depth preprocessing: " + elapsedDepth + " ms");

                    int depthWidth = tensorImageDepth.getWidth();
                    int depthHeight = tensorImageDepth.getHeight();
                    FloatBuffer depthOutput = FloatBuffer.allocate(depthHeight * depthWidth);

                    beforeDepth = System.currentTimeMillis();
                    depthAnythingModel.run(tensorImageDepth.getBuffer().asFloatBuffer(), depthOutput);
                    elapsedDepth = System.currentTimeMillis() - beforeDepth;
                    Log.i("depth", "depth inference: " + elapsedDepth + " ms");

                    depthOutput = Image.rgbImageTranspose(depthOutput, DEPTH_IMAGE_SIZE, DEPTH_IMAGE_SIZE);

                    Log.i("DEPTH", Arrays.toString(depthOutput.array()));
                    Log.i("BITMAP", Image.describeBitmap(bitmap));
                    Bitmap finalDepthBitmap = Image.greyscaleBufferToBitmap(depthOutput, depthWidth, depthHeight);
                    Log.i("BITMAP", Image.describeBitmap(finalDepthBitmap));
                    //Bitmap finalDepthBitmap = bitmap; // works
                    activity.runOnUiThread(() -> {
                        Log.i("BITMAP", "displaying depth bitmap");
                        activity.displayPhoto(finalDepthBitmap, imageRot);
                        Log.i("BITMAP", "issued command: displaying depth bitmap");
                    });
                }

                super.onCaptureSuccess(image);
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e("PHOTO", "there has been an error taking the photo");
                Log.e("PHOTO", exception.toString());
                queueSpeak("Il y a eu une erreur quand j'ai essayé de prendre une photo");
                super.onError(exception);
            }
        });
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
