package com.example.androidassistant.activities;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.os.Bundle;
import android.util.Log;
import android.util.Size;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.LifecycleOwner;

import com.example.androidassistant.AssistantApp;
import com.example.androidassistant.ModelManager;
import com.example.androidassistant.R;
import com.example.androidassistant.Sonar;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

public class SonarActivity extends AppCompatActivity {
    private static String TAG = "SonarActivity";

    private ExecutorService cameraExecutor;
    private AssistantApp app;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_sonar);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Log.i(TAG, "onCreate: starting sonar");

        app = AssistantApp.getInstance();

        // unload all models in ModelManager to maximize performance
        ModelManager modelManager = app.getModelManager();
        modelManager.unloadAll();

        Interpreter interpreter = null;
        try {
            interpreter = modelManager.getModel(app, "depth_anything").get();
        } catch (IOException e) {
            app.queueSpeak("Erreur au moment de charger le modèle pour le sonar");
            throw new RuntimeException(e);
        }

        cameraExecutor = Executors.newSingleThreadExecutor();
        startSonar(interpreter);
    }

    private void startSonar(Interpreter interpreter) {
        // Initialize the camera provider
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider, interpreter);
            } catch (ExecutionException | InterruptedException e) {
                // Handle exceptions
                Log.e(TAG, "Error getting camera provider", e);
            }
        }, ContextCompat.getMainExecutor(this));

    }

    private String formatDistance(float distance) {
        String formatted = String.format(Locale.FRANCE, "%.1f", distance);

        if (formatted.endsWith(",0")) {
            formatted = formatted.substring(0, formatted.length() - 2);
        }

        return formatted;
    }

    private  void bindPreview(ProcessCameraProvider cameraProvider, Interpreter interpreter) {

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        // enable the following line if RGBA output is needed.
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setResolutionStrategy(new ResolutionStrategy(new Size(518, 518), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                // insert your code here.

                Log.i(TAG, "analyze: test");

                float distance = Sonar.getDistance(imageProxy, interpreter);

                String formattedDistance = formatDistance(distance);

                app.blockingSpeak(formattedDistance + " mètres.");

                // after done, release the ImageProxy object
                imageProxy.close();
            }
        });

        // Select back camera as a default
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        app.interruptSpeech();
    }
}