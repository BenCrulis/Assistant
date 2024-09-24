package com.example.androidassistant.database;

public class Embedding {

    private final float[] vector;

    public Embedding(float[] vector) {
        this.vector = vector;
    }

    public float[] getVector() {
        return this.vector;
    }

}
