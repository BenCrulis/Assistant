package com.example.androidassistant.database;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.PrimaryKey;

import java.nio.ByteBuffer;

@Entity( //primaryKeys = {"savedObject", "embId"},
        foreignKeys = @ForeignKey(entity = SavedObject.class,
                                    parentColumns = "objId",
                                    childColumns = "savedObject",
                                    onDelete = ForeignKey.CASCADE))
public class ObjEmbedding {

    @ColumnInfo(index = true)
    public int savedObject;

    @PrimaryKey(autoGenerate = true)
    public int embId;

    @ColumnInfo(typeAffinity = ColumnInfo.BLOB)
    public Embedding embed;

}
