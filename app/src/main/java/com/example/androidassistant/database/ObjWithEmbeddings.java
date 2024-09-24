package com.example.androidassistant.database;

import androidx.room.Embedded;
import androidx.room.Entity;
import androidx.room.PrimaryKey;
import androidx.room.Relation;

import java.util.List;

@Entity
public class ObjWithEmbeddings {

    @PrimaryKey
    public int dummy;

//    @Embedded
//    public SavedObject savedObject;
//
//    @Relation(
//            parentColumn = "objId",
//            entityColumn = "savedObject"
//    )
//    public List<ObjEmbedding> embeddings;

}
