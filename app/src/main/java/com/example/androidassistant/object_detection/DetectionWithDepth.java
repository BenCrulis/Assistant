package com.example.androidassistant.object_detection;

public class DetectionWithDepth {

    public final String object_label;
    public final float confidence;
    public final float x;
    public final float y;
    public final float depth;

    public DetectionWithDepth(String objectLabel, float confidence, float x, float y, float depth) {
        object_label = objectLabel;
        this.confidence = confidence;
        this.x = x;
        this.y = y;
        this.depth = depth;
    }
}
