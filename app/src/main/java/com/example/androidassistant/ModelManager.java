package com.example.androidassistant;

import static androidx.core.content.ContextCompat.getDataDir;

import android.app.Application;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class ModelManager {

    private static class ModelWithTimestamp {
        public final Model model;
        public Instant timestamp;

        public ModelWithTimestamp(Model model, Instant timestamp) {
            this.model = model;
            this.timestamp = timestamp;
        }

        public static ModelWithTimestamp create(Model model) {
            return new ModelWithTimestamp(model, Instant.now());
        }

        public Model accessModel() {
            this.timestamp = Instant.now();
            return this.model;
        }
    }

    private static class ModelFileAndSize {
        public final String file;
        public final int size;

        public ModelFileAndSize(String file, int size) {
            this.file = file;
            this.size = size;
        }
    }

    private int maxMemory;
    private Application application;
    private String basePath;

    private final HashMap<String, ModelFileAndSize> registeredModels;
    private final HashMap<String, ModelWithTimestamp> loadedModels;


    public ModelManager(Application application, int maxMemory) {
        this.maxMemory = maxMemory;
        this.registeredModels = new HashMap<>();
        this.loadedModels = new HashMap<>();
        this.application = application;
        basePath = application.getDataDir().getAbsolutePath() + "/";
    }

    public int getMaxMemory() {
        return maxMemory;
    }

    public void setMaxMemory(int maxMemory) {
        if (maxMemory <= 0) {
            throw new RuntimeException("Cannot set max memory to a negative number or 0");
        }
        this.maxMemory = maxMemory;
    }

    public void registerModel(String name, String filename, int size) {
        logInfo("Registering model: " + name);
        ModelFileAndSize matchingRegistered = this.registeredModels.get(name);
        if (matchingRegistered != null) {
            if (!matchingRegistered.file.equals(filename)) {
                throw new RuntimeException("Model " + name + " is already registered with file " + matchingRegistered.file);
            }
            if (matchingRegistered.size != size) {
                throw new RuntimeException("Model " + name + " is already registered with size " + matchingRegistered.size);
            }
        }

        this.registeredModels.put(name, new ModelFileAndSize(filename, size));
    }

    public void registerModel(Application app, String name, String filename) {
        logInfo("Reading the size of the model file: " + filename);
        int size = -1;
        try (InputStream inputStream = app.getAssets().open(filename);) {
            size = inputStream.available();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        registerModel(name, filename, size);
    }

    public int totalAllocated() {
        int allocated = 0;
        for (Map.Entry<String, ModelWithTimestamp> entry : loadedModels.entrySet()) {
            int modelSize = entry.getValue().model.getSize();
            allocated += modelSize;
        }
        return allocated;
    }

    public void unloadModel(String name) {
        ModelWithTimestamp modelWithTimestamp = this.loadedModels.get(name);
        if (modelWithTimestamp != null) {
            try {
                modelWithTimestamp.model.close();
                this.loadedModels.remove(name);
                logInfo("Unloaded model: " + name);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void attemptDeallocationForSize(int size) {
        ArrayList<Map.Entry<String, ModelWithTimestamp>> model = new ArrayList<>(this.loadedModels.entrySet());
        model.sort(Comparator.comparingLong(x -> x.getValue().timestamp.getEpochSecond()));

        logInfo("starting deallocation to regain " + size + " bytes");

        int runningSize = 0;
        int i = 0;
        while (runningSize < size && i < model.size()) {
            runningSize += model.get(i).getValue().model.getSize();
            i++;
        }

        ArrayList<String> entriesToRemove = new ArrayList<>();

        for (int j = 0; j < i; j++) {
            Map.Entry<String, ModelWithTimestamp> entry = model.get(j);
            entriesToRemove.add(entry.getKey());
        }

        for (String key : entriesToRemove) {
            this.unloadModel(key);
        }

        logInfo("Unloaded " + i + " model(s)");
    }

    public void unloadAll() {
        attemptDeallocationForSize(this.maxMemory);
        logInfo("unloadAll: Unloaded all models.");
    }

    public Optional<Interpreter> getModel(Application app, String name) throws IOException {
        ModelFileAndSize modelFileAndSize = registeredModels.get(name);
        if (modelFileAndSize == null) {
            logInfo("Could not find registered model: " + name);
            return Optional.empty();
        }

        ModelWithTimestamp modelWithTimestamp = loadedModels.get(modelFileAndSize.file);

        if (modelWithTimestamp != null) {
            logInfo("Model " + name + " was already loaded from file "
                    + modelFileAndSize.file + ", returning it.");
            return Optional.ofNullable(modelWithTimestamp.accessModel().getInterpreter());
        }
        else {
            // attempt to load the model
            logInfo("Attempting to load model " + name + " from file " + modelFileAndSize.file);

            // first check if it is smaller than maxMemory
            if (modelFileAndSize.size > this.maxMemory) {
                logInfo("Model is too large to be loaded, limit set to " +
                        this.maxMemory + ", model size is " + modelFileAndSize.size);
                throw new RuntimeException("Model is larger than authorized memory limit of ModelManager");
            }

            int currentlyAllocated = this.totalAllocated();
            int afterNewAllocation = currentlyAllocated + modelFileAndSize.size;

            if (afterNewAllocation > this.maxMemory) {
                // need to unload one or more models
                logInfo("Need to unload one or more models");
                this.attemptDeallocationForSize(modelFileAndSize.size);
            }

            // load model
            logInfo("Loading new model");
            Model model = Model.loadFromFile(app, modelFileAndSize.file);
            ModelWithTimestamp newModelWithTimestamp = ModelWithTimestamp.create(model);
            this.loadedModels.put(modelFileAndSize.file, newModelWithTimestamp);
            logInfo("Model loaded successfully.");
            return Optional.ofNullable(model.getInterpreter());
        }
    }

    private void logInfo(String string) {
        Log.i("ModelManager", string);
    }

}
