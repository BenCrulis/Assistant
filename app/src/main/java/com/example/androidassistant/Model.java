package com.example.androidassistant;


import android.app.Application;
import android.util.Log;

import com.example.androidassistant.utils.TFLite;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

import kotlin.NotImplementedError;

public class Model implements AutoCloseable {

    private boolean closed;

    private final int size;

    private TFLite.ModelWithBuffer modelWithBuffer;

    public Model(InputStream inputStream, int size) {
        this.modelWithBuffer = TFLite.loadModelWithSizeUnsafe(inputStream, size);
        this.size = size;
        this.closed = false;
    }

    public static Model loadFromAssets(Application app, String filename) throws IOException {
        InputStream inputStream = app.getAssets().open(filename);
        int size = inputStream.available();
        Model model = new Model(inputStream, size);
        inputStream.close();
        return model;
    }

    public static Model loadFromFile(Application app, String filename) throws IOException {
        InputStream inputStream = Files.newInputStream(app.getDataDir().toPath().resolve(filename));
        int size = inputStream.available();
        Model model = new Model(inputStream, size);
        inputStream.close();
        return model;
    }

    public Interpreter getInterpreter() {
        return this.modelWithBuffer.interpreter;
    }

    public int getSize() {
        return this.size;
    }

    @Override
    public void close() throws Exception {
        if (closed) {
            throw new Exception("Model was closed");
        }

        modelWithBuffer.unsafeDeallocate();
        modelWithBuffer = null;
        closed = true;
    }
}
