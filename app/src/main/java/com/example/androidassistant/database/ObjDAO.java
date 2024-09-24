package com.example.androidassistant.database;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Transaction;

import java.nio.ByteBuffer;
import java.util.List;

@Dao
public interface ObjDAO {

    @Query("SELECT * FROM savedobject")
    List<SavedObject> getAll();

//    @Transaction
//    @Query("SELECT * FROM SavedObject")
//    List<ObjWithEmbeddings> getObjWithEmbeddings();

    @Query("SELECT * FROM savedobject where objId = :id")
    SavedObject getFromId(int id);

    @Query("SELECT max(objId) FROM savedobject")
    int getMaxId();

    @Insert
    void insertAll(SavedObject... savedObject);

    @Delete
    void deleteSavedObject(SavedObject savedObject);

    @Query("DELETE FROM savedobject where objId = :id")
    void deleteObjWithId(int id);

    @Query("UPDATE savedobject SET objName = :newName WHERE objId = :id")
    void renameObject(int id, String newName);

    @Query("INSERT INTO objembedding (savedObject, embed) VALUES (:id, :embed)")
    void addEmbedding(int id, Embedding embed);

    @Query("SELECT count(*) FROM objembedding WHERE savedObject = :id")
    int numEmbeddingsForObject(int id);

    @Query("SELECT * FROM objembedding WHERE savedObject = :id")
    List<ObjEmbedding> getEmbeddingsForObject(int id);

    @Query("SELECT savedObject FROM objembedding GROUP BY savedObject HAVING count(savedObject) > 0")
    List<Integer> ObjectIdsWithAtLeastOneEmb();

}
