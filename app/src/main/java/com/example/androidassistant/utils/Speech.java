package com.example.androidassistant.utils;

import android.speech.tts.TextToSpeech;
import android.util.Log;

public class Speech {

    private TextToSpeech tts;

    public Speech(TextToSpeech tts) {
        this.tts = tts;
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
