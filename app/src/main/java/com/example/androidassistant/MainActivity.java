package com.example.androidassistant;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraProvider;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;

import android.Manifest;

import android.content.Intent;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.*;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import org.tensorflow.lite.*;


public class MainActivity extends AppCompatActivity {
    TextToSpeech tts = null;
    CountDownLatch ttsCountDownLatch = null;

    ImageCapture imageCapture = null;
    private static final int MY_CAMERA_REQUEST_CODE = 100; // Any unique integer
    static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int SPEECH_REQUEST_CODE = 0;

    private final ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build();

    private Interpreter interpreter = null;


    // Create an intent that can start the Speech Recognizer activity
    private void displaySpeechRecognizer() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        // This starts the activity and populates the intent with the speech text.
        startActivityForResult(intent, SPEECH_REQUEST_CODE);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode,
                                    Intent data) {
        if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK) {
            List<String> results = data.getStringArrayListExtra(
                    RecognizerIntent.EXTRA_RESULTS);
            String spokenText = results.get(0);
            // Do something with spokenText.
            queueSpeak(spokenText);
        }
        super.onActivityResult(requestCode, resultCode, data);
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
            Button voiceButton = findViewById(R.id.voice_input);
            voiceButton.setEnabled(true);
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

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            // Check if the permission was granted or denied
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i("CAMERA", "permission was granted");
            } else {
                Log.i("CAMERA", "permission was denied");
            }
        }
    }


    public void displayPhoto(Bitmap bitmap) {
        Toast toast = new Toast(this);
        ImageView view = new ImageView(this);
        view.setImageBitmap(bitmap);
        toast.setView(view);
        toast.show();
    }

    public void takePhoto() {
        MainActivity activity = this;
        imageCapture.takePicture(Executors.newSingleThreadExecutor(),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        Log.i("PHOTO", "photo successfully taken");
//                        queueSpeak("J'ai pris une photo, j'analyse...");
                        int height = image.getHeight();
                        int width = image.getWidth();
                        int planes = image.getPlanes().length;
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
                        Log.i("buffer", Arrays.toString(intArray));

                        // Create a TensorImage object. This creates the tensor of the corresponding
                        // tensor type that the TensorFlow Lite interpreter needs.
                        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

                        // Analysis code for every frame
                        // Preprocess the image
                        tensorImage.load(bitmap);
                        tensorImage = imageProcessor.process(tensorImage);

                        FloatBuffer output = FloatBuffer.allocate(1000);

                        interpreter.run(tensorImage.getBuffer().asFloatBuffer(), output);

                        float[] outputArray = output.array();

                        int maxAt = 0;

                        for (int i = 0; i < outputArray.length; i++) {
                            maxAt = outputArray[i] > outputArray[maxAt] ? i : maxAt;
                        }

                        queueSpeak("Classe " + maxAt);

                        Bitmap finalBitmap = bitmap;
                        activity.runOnUiThread(() -> {
                            displayPhoto(finalBitmap);
                        });

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // setup camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

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
                .setTargetRotation(windowManager.getDefaultDisplay().getRotation())
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


        cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                imageCapture);

        //this.displaySpeechRecognizer();

        // TFLite part
        Interpreter.Options options = new Interpreter.Options()
                .setUseNNAPI(false)
                .setNumThreads(1);

        String basePath = getDataDir().getAbsolutePath() + "/";

        String modelName = "quicknet.tflite"; // input/output in float
//        String modelName = "mobilenetv1.tflite"; // input/output in byte

        File modelFile = new File(basePath + modelName);
        Log.i("info", "opening asset model " + modelName);
        try (InputStream assetStream = getAssets().open(modelName)) {
            Log.i("info", "copying asset model");
            Files.copy(assetStream, Paths.get(basePath + modelName), StandardCopyOption.REPLACE_EXISTING);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        // setup buttons
        Button button = (Button) findViewById(R.id.classify_photo);
        button.setOnClickListener(v -> {
            Log.d("BUTTONS", "Taking photo and classifying it");
            tts.speak("Je prend une photo et je la classifie", TextToSpeech.QUEUE_FLUSH, null, null);
            Log.i("TTS", "started speaking");
            takePhoto();
        });


        Button voiceReconButton = findViewById(R.id.voice_input);
        voiceReconButton.setOnClickListener(v -> {
            Log.i("VOICE", "starting voice recognition");
            this.displaySpeechRecognizer();
        });

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
            FloatBuffer inputs = FloatBuffer.allocate(224*224*3);
            Arrays.fill(inputs.array(), 1.0f);
            inputs.rewind();
            FloatBuffer outputs = FloatBuffer.allocate(1000);
//            interpreter.resizeInput(0, new int[]{3*224*224});
            interpreter.run(inputs, outputs);

            Log.i("info", Arrays.toString(outputs.array()));
        }
        catch (Exception e) {
            Log.e("MODEL", e.toString());
        }
    }

    public void onPause(){
//        if(tts !=null){
//            tts.stop();
//            tts.shutdown();
//        }
        super.onPause();
    }

    public void onDestroy() {
        if(tts !=null){
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}