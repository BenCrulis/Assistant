package com.example.androidassistant.utils;

import java.util.Arrays;
import java.util.Comparator;

public class ArrayUtils {

    public static int argmax(final float[] array) {
        int idx = -1;
        float max_val = Float.MIN_VALUE;

        for (int i = 0; i < array.length; i++) {
            float val = array[i];
            if (val > max_val) {
                idx = i;
                max_val = val;
            }
        }
        return idx;
    }

    public static int[] argsort(final float[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, (i1, i2) -> (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]));
        return asArray(indexes);
    }

    @SafeVarargs
    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }

}
