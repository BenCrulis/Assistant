package com.example.androidassistant.utils;

import static androidx.core.content.PermissionChecker.checkCallingOrSelfPermission;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.androidassistant.AssistantApp;

import java.util.ArrayList;
import java.util.function.Consumer;

public class SpeechRecognizerUtils {

    public static void requestRecordAudioPermission(Activity activity) {
        Log.i("SPEECH", "checking record audio permission");
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.i("SPEECH", "record permission not granted, asking for it");
            ActivityCompat.requestPermissions(activity, new String[]{Manifest.permission.RECORD_AUDIO}, RequestCodes.SPEECH_REQUEST);
        }
    }

    public static void recognizeSpeech(Context context, Consumer<ArrayList<String>> callback) {

        SpeechRecognizer speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context);

        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle bundle) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float v) {

            }

            @Override
            public void onBufferReceived(byte[] bytes) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            public void finish() {
                speechRecognizer.stopListening();
                speechRecognizer.destroy();
            }

            @Override
            public void onError(int i) {
                Log.e("SPEECH", "there has been an error in speech recognition: " + i);

                if (i == SpeechRecognizer.ERROR_NO_MATCH) {
                    AssistantApp.getInstance().queueSpeak("Je n'ai pas compris.");
                } else if (i == SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) {
                    AssistantApp.getInstance().queueSpeak(
                            "Je n'ai pas la permission d'utiliser le microphone." +
                            "Veuillez redémarrer l'application et accepter la demande de permission.");
                }
                else if (i == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE) {
                    AssistantApp.getInstance().queueSpeak(
                            "La langue française n'est pas supportée par votre appareil."
                    );
                }
                else {
                    AssistantApp.getInstance().queueSpeak(
                            "Erreur numéro " + i + " lors de l'utilisation de la reconnaissance vocale. Veuillez rapporter cette erreur au développeur.");
                }
            }

            @Override
            public void onResults(Bundle bundle) {
                callback.accept(bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION));
                this.finish();
            }

            @Override
            public void onPartialResults(Bundle bundle) {

            }

            @Override
            public void onEvent(int i, Bundle bundle) {

            }
        });

        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
//        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "fr-FR");
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.getPackageName());
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "parlez maintenant");
        intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true);

        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.getPackageName());
        // This starts the activity and populates the intent with the speech text.
        speechRecognizer.startListening(intent);
    }

}
