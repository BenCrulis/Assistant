package com.example.androidassistant.database;

import androidx.room.Database;
import androidx.room.RoomDatabase;
import androidx.room.TypeConverters;

@Database(entities = {SavedObject.class, ObjEmbedding.class, ObjWithEmbeddings.class}, version = 1)
@TypeConverters({Converters.class})
public abstract class AppDatabase extends RoomDatabase {
    public abstract ObjDAO objDAO();
}