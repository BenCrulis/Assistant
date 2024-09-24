package com.example.androidassistant.database;

import androidx.room.TypeConverter;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class Converters {
    @TypeConverter
    public static Embedding fromBlob(byte[] blob) {
        assert blob.length % 4 == 0;
        ByteBuffer buffer = ByteBuffer.allocate(blob.length);
        buffer.put(blob);
        buffer.rewind();

        float floats[] = new float[blob.length / 4];
        buffer.asFloatBuffer().get(floats, 0, floats.length);

        return new Embedding(floats);
    }

    @TypeConverter
    public static byte[] toBlob(Embedding embedding) {
        float[] floats = embedding.getVector();
        ByteBuffer buffer = ByteBuffer.allocateDirect(floats.length * 4);
        buffer.asFloatBuffer().put(floats);

        byte[] bytes = new byte[floats.length * 4];
        buffer.get(bytes, 0, bytes.length);

        return bytes;
    }
}
