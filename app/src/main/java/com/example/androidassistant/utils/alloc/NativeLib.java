package com.example.androidassistant.utils.alloc;

public class NativeLib {
    static {
        System.loadLibrary("native-lib");
    }

    public native void allocateMemory(int size);
    public native void freeMemory();
}
