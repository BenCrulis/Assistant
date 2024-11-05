package com.example.androidassistant;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.PackageManager;;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;

import android.Manifest;

import android.content.Intent;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import com.example.androidassistant.activities.ObjectSavingActivity;
import com.example.androidassistant.activities.SonarActivity;
import com.example.androidassistant.utils.RequestCodes;
import com.example.androidassistant.utils.SpeechRecognizerUtils;
import com.google.android.material.button.MaterialButton;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class MainActivity extends AppCompatActivity {
    AssistantApp assistantApp;
    private static final int MY_CAMERA_REQUEST_CODE = 100; // Any unique integer
    private static final int SPEECH_REQUEST_CODE = 0;

    String[] permissionStr = {
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA,
    };

    ActivityResultLauncher<String[]> permissionsLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(),
                    new ActivityResultCallback<Map<String, Boolean>>() {
                        @Override
                        public void onActivityResult(Map<String,Boolean> result) {

                        }
                    });

    private AssistantApp getAssistantApp() {
        return AssistantApp.getInstance();
    }

    private void askForPermissions(List<String> permissionsList) {
        String[] newPermissionStr = new String[permissionsList.size()];
        for (int i = 0; i < newPermissionStr.length; i++) {
            newPermissionStr[i] = permissionsList.get(i);
        }
        if (newPermissionStr.length > 0) {
            permissionsLauncher.launch(newPermissionStr);
        } else {
        /* User has pressed 'Deny & Don't ask again' so we have to show the enable permissions dialog
        which will lead them to app details page to enable permissions from there. */
            showPermissionDialog();
        }
    }

    AlertDialog alertDialog;private void showPermissionDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Autorisation nécessaire")
                .setMessage("Des permissions sont nécessaire pour le bon fonctionnement de l'application.")
                .setPositiveButton("Paramètres", (dialog, which) -> {
                    dialog.dismiss();
                });
        if (alertDialog == null) {
            alertDialog = builder.create();
            if (!alertDialog.isShowing()) {
                alertDialog.show();
            }
        }
    }

    // Create an intent that can start the Speech Recognizer activity
    private void displaySpeechRecognizer() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        // This starts the activity and populates the intent with the speech text.
        startActivityForResult(intent, SPEECH_REQUEST_CODE);
    }

    private void queueSpeak(String speech) {
        assistantApp.queueSpeak(speech);
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
        } else if (requestCode == RequestCodes.SPEECH_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i("SPEECH", "permission was granted");
            } else {
                Log.i("SPEECH", "permission was denied");
            }
        }
    }


    public void displayPhoto(Bitmap bitmap, int rotation) {
        Toast toast = new Toast(this);
        ImageView view = new ImageView(this);
        view.setImageBitmap(bitmap);
        view.setRotation((float) rotation);
        toast.setView(view);
        toast.setDuration(Toast.LENGTH_LONG);
        toast.show();
    }

    public void takePhoto(int rot) {
        assistantApp.takePhotoAndDetect(this, rot);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        List<String> permissions = Arrays.asList(permissionStr);
        askForPermissions(permissions);

        // setup camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

        SpeechRecognizerUtils.requestRecordAudioPermission(this);

        assistantApp = getAssistantApp();
        init_all();
    }

    private void init_debug() {

        LinearLayout linearLayout = (LinearLayout) findViewById(R.id.linear_button_layout);

//        Button vllm_button = (Button) findViewById(R.id.test_vllm);
        Button vllm_button = new MaterialButton(this);
        vllm_button.setText("test VLLM");
        vllm_button.setEnabled(true);
        linearLayout.addView(vllm_button);

        vllm_button.setOnClickListener(v -> {
            Log.d("BUTTONS", "Testing VLLM");
            VLLM.testMoondream(assistantApp);
        });

        Button tokenizer_button = new MaterialButton(this);
        tokenizer_button.setText("test Tokenizer");
        tokenizer_button.setEnabled(true);
        linearLayout.addView(tokenizer_button);

        tokenizer_button.setOnClickListener(v -> {
            Log.d("BUTTONS", "Testing Tokenizer");
            VLLM.testTokenizer(assistantApp);
        });

//        Button voiceReconButton = findViewById(R.id.voice_input);
        Button voiceReconButton = new MaterialButton(this);
        voiceReconButton.setText("Tester vocal");
        linearLayout.addView(voiceReconButton);

        voiceReconButton.setOnClickListener(v -> {
            Log.i("VOICE", "starting voice recognition");
            this.displaySpeechRecognizer();
        });

//        Button test_mem_button = findViewById(R.id.test_mem_button);
        Button test_mem_button = new MaterialButton(this);
        test_mem_button.setText("Test memory");
        linearLayout.addView(test_mem_button);

        test_mem_button.setOnClickListener(v -> AssistantApp.test_memory());

//        Button benchmark_button = findViewById(R.id.benchmark_button);
        Button benchmark_button = new MaterialButton(this);
        benchmark_button.setText("Benchmark models");
        linearLayout.addView(benchmark_button);

        benchmark_button.setOnClickListener(v -> AssistantApp.getInstance().benchmark_models());

//        Button testMM_button = findViewById(R.id.test_mm_button);
        Button testMM_button = new MaterialButton(this);
        testMM_button.setText("Test model manager");
        linearLayout.addView(testMM_button);
        testMM_button.setOnClickListener(v -> AssistantApp.getInstance().test_model_manager());

        Button testDbButton = new MaterialButton(this);
        testDbButton.setText("Test Database");
        linearLayout.addView(testDbButton);
        testDbButton.setOnClickListener(v -> AssistantApp.getInstance().testDb());

    }

    protected void init_all() {
        Log.i("INIT", "initializing MainActivity");

        if (BuildConfig.DEBUG) {
            Log.i("INIT", "RUNNING IN DEBUG MODE");
        }
        else {
            Log.i("INIT", "RUNNING IN RELEASE MODE");
        }

        // setup buttons
        Button button = (Button) findViewById(R.id.classify_photo);
        button.setOnClickListener(v -> {
            Log.d("BUTTONS", "Taking photo and classifying it");
            assistantApp.flushSpeak("Je prend une photo.");
            Log.i("TTS", "started speaking");

            WindowManager windowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
            int rot = windowManager.getDefaultDisplay().getRotation();
            takePhoto(rot);
        });

        Button object_saving_button = findViewById(R.id.object_saving_button);
        object_saving_button.setOnClickListener(v -> {
            assistantApp.queueSpeak("Basculement vers les objets enregistrés.");
            Intent intent = new Intent(this, ObjectSavingActivity.class);
            startActivity(intent);
        });

        Button object_recognition_button = findViewById(R.id.obj_recognition_button);
        object_recognition_button.setOnClickListener(v -> {
            assistantApp.recognize_saved_objects(this);
        });

        Button dirtiness_detection_button = findViewById(R.id.dirtiness_detection_button);
        dirtiness_detection_button.setOnClickListener(v -> {
            DirtinessDetection.detectDirtiness(assistantApp, this);
        });

        Button describe_photo_button = findViewById(R.id.describe_photo_button);
        describe_photo_button.setOnClickListener(v -> {
            assistantApp.takePhotoAndDescribe(this);
        });

        Button sonar_button = findViewById(R.id.sonar_button);
        sonar_button.setOnClickListener(v -> {
            Log.i("SONAR", "starting sonar activity");
            assistantApp.queueSpeak("Basculement vers l'estimation de distance.");
            Intent intent = new Intent(this, SonarActivity.class);
            startActivity(intent);
        });

        if (BuildConfig.SHOW_DEBUG_BUTTONS) {
            init_debug();
        }

    }

}