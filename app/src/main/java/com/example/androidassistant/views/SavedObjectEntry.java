// CustomTextButtonView.java
package com.example.androidassistant.views;

import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewParent;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.androidassistant.AssistantApp;
import com.example.androidassistant.CLIP;
import com.example.androidassistant.ModelManager;
import com.example.androidassistant.R;
import com.example.androidassistant.activities.ObjectSavingActivity;
import com.example.androidassistant.database.Embedding;
import com.example.androidassistant.database.ObjDAO;
import com.example.androidassistant.utils.SpeechRecognizerUtils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.sql.Blob;
import java.util.Arrays;


public class SavedObjectEntry extends LinearLayout {

    public static final String TAG = "SavedObjEntry";

    private TextView nameTextView;
    private TextView countTextView;
    private Button renameButton;
    private Button takePhotoButton;
    private Button deleteButton;

    private int numPhotos;

    private int objId = -1;

    private ObjDAO objDAO;

    public SavedObjectEntry(Context context) {
        super(context);
        init(context);
    }

    public SavedObjectEntry(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    public SavedObjectEntry(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init(context);
    }

    private void init(Context context) {
        // Inflate the layout
        LayoutInflater.from(context).inflate(R.layout.saved_object_entry, this, true);

        // Initialize views
        nameTextView = findViewById(R.id.obj_name);
        countTextView = findViewById(R.id.obj_count);
        renameButton = findViewById(R.id.rename_button);
        takePhotoButton = findViewById(R.id.take_photo_button);
        deleteButton = findViewById(R.id.delete_button);

        numPhotos = 0;

        objDAO = AssistantApp.getInstance().getDb().objDAO();

        deleteButton.setOnClickListener(v -> this.deleteEntry());
        takePhotoButton.setOnClickListener(v -> this.takeAndProcessPhoto());
        renameButton.setOnClickListener(v -> this.askUserForNewName());
    }

    public void setObjId(int objId) {
        this.objId = objId;
    }

    public int getObjId() {
        return objId;
    }

    public String getLabel() {
        return nameTextView.getText().toString();
    }

    // Method to set label text
    public void setLabelText(String text) {
        nameTextView.setText(text);
    }

    // Method to set button click listener
    public void setDeleteButtonClickListener(OnClickListener listener) {
        deleteButton.setOnClickListener(listener);
    }

    private ObjectSavingActivity getObjectSavingActivityFromView() {
        Context context = getContext();
        // Traverse up the context chain if the context is not an activity
        while (context instanceof android.content.ContextWrapper) {
            if (context instanceof ObjectSavingActivity) {
                return (ObjectSavingActivity) context;
            }
            context = ((android.content.ContextWrapper) context).getBaseContext();
        }
        return null;
    }

    public void deleteEntry() {
        ObjectSavingActivity objectSavingActivity = getObjectSavingActivityFromView();

        if (objectSavingActivity != null) {
            objectSavingActivity.removeEntry(this);
            return;
        }

        View root = getRootView();
        LinearLayout linearLayout = root.findViewById(R.id.all_objects_layout);
        linearLayout.removeView(this);
    }

    public void setNumPhotos(int numPhotos) {
        if (numPhotos < 0) {
            throw new IllegalArgumentException("number of photos must be positive or null");
        }
        countTextView.setText("" + numPhotos + " photos");
    }

    public int getNumPhotos() {
        return numPhotos;
    }


    private void takeAndProcessPhoto() {
        Log.i("ObjEntry", "taking photo for object: " + getLabel());
        AssistantApp assistantApp = AssistantApp.getInstance();

        ObjectSavingActivity objectSavingActivity = this.getObjectSavingActivityFromView();
        assistantApp.takePhoto(objectSavingActivity, tensorImage -> {
            Log.i("ObjEntry", "successfully got tensorImage object");

            ModelManager modelManager = assistantApp.getModelManager();

            CLIP clip = new CLIP(modelManager);

            TensorBuffer emb = clip.infer(tensorImage);

            Log.i("ObjEntry", "shape: " + Arrays.toString(emb.getShape()));
            Log.i("ObjEntry", "emb[:10]: " +
                    Arrays.toString(Arrays.copyOfRange(emb.getFloatArray(), 0, 10)));

            float[] floatEmb = emb.getFloatArray();
            assert floatEmb.length == 512;

            // process embedding with whitening and coloring
            Interpreter wc;
            try {
                wc = modelManager.getModel(assistantApp, "WC").get();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            emb.getBuffer();

            FloatBuffer wc_out = FloatBuffer.allocate(512);
            wc.run(emb.getBuffer().rewind(), wc_out);

            floatEmb = wc_out.array();

            objDAO.addEmbedding(this.getObjId(), new Embedding(floatEmb));
            Log.i("ObjEntry", "added embedding to database for object: " + this.getObjId());

            int numPhotos = objDAO.numEmbeddingsForObject(this.getObjId());

            this.setNumPhotos(numPhotos);

            Log.i("ObjEntry", "Object of id " + this.getObjId() + " has " + this.getNumPhotos() + " photo(s)");

            objectSavingActivity.runOnUiThread(objectSavingActivity::refreshDisplay);

            AssistantApp.getInstance().queueSpeak("L'objet \"" + this.getLabel() + "\" a maintenant " + numPhotos + " photos.");
        });
    }

    public void askUserForNewName() {
        ObjectSavingActivity objectSavingActivity = this.getObjectSavingActivityFromView();

        AssistantApp app = AssistantApp.getInstance();

        Log.i(TAG, "askUserForNewName: will start speech recognition to get new name");

        app.blockingSpeak("Veuillez prononcer le nouveau nom de l'objet.");

        SpeechRecognizerUtils.recognizeSpeech(this.getContext(), lines -> {
            Log.i(TAG, "askUserForNewName: speech recognized:");
            for (String line: lines) {
                Log.i(TAG, "askUserForNewName: " + line);
            }
            Log.i(TAG, "askUserForNewName: using first line as new name if not empty");

            if (lines.size() <= 0) {
                // todo: handle no line recognized
                app.queueSpeak("Je n'ai pas entendu de nouveau nom, abandon.");
                return;
            }

            String oldName = this.getLabel();
            String newName = lines.get(0);

            if (newName.isEmpty()) {
                // todo: handle no text
                app.queueSpeak("Le nom donné est vide, abandon.");
                return;
            }

            // todo: handle new name already exists
            objDAO.renameObject(this.getObjId(), newName);
            Log.i(TAG, "askUserForNewName: renamed object to: " + newName);
            app.queueSpeak("L'objet \"" + oldName + "\" s'appelle maintenant: \"" + newName + "\"");
            objectSavingActivity.runOnUiThread(objectSavingActivity::refreshDisplay);
        });
    }

}
