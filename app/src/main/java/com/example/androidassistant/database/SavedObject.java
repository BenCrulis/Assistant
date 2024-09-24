package com.example.androidassistant.database;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity
public class SavedObject {

    @PrimaryKey(autoGenerate = true)
    public int objId;

    public String objName;



}
