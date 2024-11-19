package com.example.androidassistant.activities;

import android.os.Bundle;

import com.example.androidassistant.AssistantApp;
import com.example.androidassistant.database.ObjDAO;
import com.example.androidassistant.database.SavedObject;
import com.example.androidassistant.views.SavedObjectEntry;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.androidassistant.databinding.SavedObjectsDisplayBinding;

import com.example.androidassistant.R;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ObjectSavingActivity extends AppCompatActivity {

    private SavedObjectsDisplayBinding binding;

    private LinearLayout all_objects_layout;
    private TextView numObjTextView;

    private ObjDAO objDAO;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = SavedObjectsDisplayBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        objDAO = AssistantApp.getInstance().getDb().objDAO();

        Button add_obj_button = findViewById(R.id.add_obj_button);
        add_obj_button.setOnClickListener(v -> this.addEntry());

        numObjTextView = findViewById(R.id.numObjTextView);

        all_objects_layout = findViewById(R.id.all_objects_layout);

//        SavedObjectEntry entry = new SavedObjectEntry(this);
//        all_objects_layout.addView(entry);
//        updateNumObjTextView();

        displayObjectEntriesFromDb();

    }

    public SavedObjectEntry objEntryFromSavedObject(SavedObject savedObject) {
        SavedObjectEntry entry = new SavedObjectEntry(this);
        entry.setLabelText(savedObject.objName);
        entry.setObjId(savedObject.objId);
        int numEmbeddings = objDAO.numEmbeddingsForObject(savedObject.objId);
        entry.setNumPhotos(numEmbeddings);
        return entry;
    }

    public void displayObjectEntriesFromDb() {
        for (SavedObject savedObject : objDAO.getAll()) {
            SavedObjectEntry entry = objEntryFromSavedObject(savedObject);
            all_objects_layout.addView(entry);
        }
        updateNumObjTextView();
    }

    public void refreshDisplay() {
        refreshEntryDisplay();
        updateNumObjTextView();
    }

    public void refreshEntryDisplay() {
        for (SavedObjectEntry entry : getAllObjectEntries()) {
            all_objects_layout.removeView(entry);
        }
        displayObjectEntriesFromDb();
    }

    private ArrayList<SavedObjectEntry> getAllObjectEntries() {
        ArrayList<SavedObjectEntry> entries = new ArrayList<>();
        for (int i = 0; i < all_objects_layout.getChildCount(); i++) {
            View view = all_objects_layout.getChildAt(i);
            if (view instanceof SavedObjectEntry) {
                SavedObjectEntry entry = (SavedObjectEntry) view;
                entries.add(entry);
            }
        }
        return entries;
    }

    public int getMaxUnnamedObjectNumber() {
        int maximum = 0;

        Pattern unnamedObjRegex = Pattern.compile("Objet ([0-9]+)");

        for (int i = 0; i < all_objects_layout.getChildCount(); i++) {
            View child = all_objects_layout.getChildAt(i);
            if (child instanceof SavedObjectEntry) {
                SavedObjectEntry entry = (SavedObjectEntry) child;
                String label = entry.getLabel();
                Matcher matcher = unnamedObjRegex.matcher(label);
                if (matcher.matches()) {
                    String stringNumber = matcher.group(1);
                    assert stringNumber != null;
                    int number = Integer.parseInt(stringNumber);
                    maximum = Math.max(maximum, number);
                }
            }
        }
        return maximum;
    }

    public int getNumObjWithAtLeastOnePhoto() {
        int c = objDAO.ObjectIdsWithAtLeastOneEmb().size();
        Log.i("ObjSaving", "number of objects with photo (DAO): " + c);

        return c;
    }

    public void updateNumObjTextView() {
        int numObj = all_objects_layout.getChildCount();
        int numObjWithPhoto = this.getNumObjWithAtLeastOnePhoto();
        String txt = "" + numObj  + " objets enregistrés dont " + numObjWithPhoto + " avec au moins une photo";
        this.numObjTextView.setText(txt);
    }

    protected void addEntry() {
        int maxId = objDAO.getMaxId();
        int nextID = maxId + 1;

        SavedObject savedObject = new SavedObject();
        savedObject.objId = nextID;
        savedObject.objName = "Objet " + savedObject.objId;

        objDAO.insertAll(savedObject);

        SavedObjectEntry entry = objEntryFromSavedObject(savedObject);

        all_objects_layout.addView(entry);
        updateNumObjTextView();
    }

    public void removeEntry(SavedObjectEntry entry) {
        Log.i("ObjectSaving", "will delete entry id: " + entry.getObjId());
        objDAO.deleteObjWithId(entry.getObjId());

        all_objects_layout.removeView(entry);
        updateNumObjTextView();
    }

}