package com.example.androidassistant.utils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.example.androidassistant.AssistantApp;

public class Debug {

    public static Bitmap getSceneImage(AssistantApp app) {
        String basePath = app.getDataDir().getAbsolutePath() + "/";

        // load scene image
        Bitmap diskImage = BitmapFactory.decodeFile(basePath + "gmu_scene_2_rgb_462.png");

        return diskImage;
    }

}
