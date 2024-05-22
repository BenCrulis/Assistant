package com.example.androidassistant;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageCapture;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.util.Log;
import android.widget.Button;

import android.Manifest;

import android.content.Intent;
import android.widget.ImageView;
import android.widget.Toast;

import java.util.List;


public class MainActivity extends AppCompatActivity {
    AssistantApp assistantApp;
    private static final int MY_CAMERA_REQUEST_CODE = 100; // Any unique integer
    private static final int SPEECH_REQUEST_CODE = 0;

    private AssistantApp getAssistantApp() {
        return (AssistantApp) getApplication();
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

    public void takePhoto() {
        assistantApp.takePhoto(this);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // setup camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

        assistantApp = getAssistantApp();
        init_all();
    }

    protected void init_all() {
        Log.i("INIT", "initializing MainActivity");

        // setup buttons
        Button button = (Button) findViewById(R.id.classify_photo);
        button.setOnClickListener(v -> {
            Log.d("BUTTONS", "Taking photo and classifying it");
            assistantApp.flushSpeak("Je prend une photo.");
            Log.i("TTS", "started speaking");
            takePhoto();
        });


        Button voiceReconButton = findViewById(R.id.voice_input);
        voiceReconButton.setOnClickListener(v -> {
            Log.i("VOICE", "starting voice recognition");
            this.displaySpeechRecognizer();
        });

    }

}