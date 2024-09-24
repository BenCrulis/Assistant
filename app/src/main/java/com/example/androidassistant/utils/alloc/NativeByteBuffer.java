package com.example.androidassistant.utils.alloc;

import androidx.annotation.NonNull;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

public class NativeByteBuffer {
    static {
        System.loadLibrary("native-lib");
    }

    public static native ByteBuffer createBuffer(int size);
    public static native void deallocateBuffer(ByteBuffer buffer);

}
