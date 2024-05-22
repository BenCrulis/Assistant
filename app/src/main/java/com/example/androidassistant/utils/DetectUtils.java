package com.example.androidassistant.utils;

import android.util.Pair;

import com.example.androidassistant.YOLOW;

import org.jetbrains.kotlinx.multik.api.Engine;
import org.jetbrains.kotlinx.multik.api.EngineKt;
import org.jetbrains.kotlinx.multik.api.KEEngineType;
import org.jetbrains.kotlinx.multik.ndarray.data.D1;
import org.jetbrains.kotlinx.multik.ndarray.data.D2;
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray;
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray;
import org.jetbrains.kotlinx.multik.ndarray.data.SliceKt;
import org.jetbrains.kotlinx.multik.ndarray.data.ViewGettersAndSettersKt;
import org.jetbrains.kotlinx.multik.ndarray.data.SliceEndStub;
import org.tensorflow.lite.support.image.BoundingBoxUtil;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;


public class DetectUtils {
    public static final Engine engine = EngineKt.enginesProvider().get(KEEngineType.INSTANCE);
    public static final org.jetbrains.kotlinx.multik.api.math.Math math = YOLOW.engine.getMath();

    public static class Detect {
        public final int class_idx;
        public final float probability;
        public final float[] bbox;

        public String label;

        public Detect(int class_idx, float probability, float[] bbox) {
            this.class_idx = class_idx;
            this.probability = probability;
            this.bbox = bbox;
            this.label = null;
        }

        @Override
        public String toString() {
            String class_label_str = "";
            if (this.label != null) {
                class_label_str = " " + this.label + ", ";
            }
            return "Detect{" + class_label_str +
                    "class_idx=" + class_idx +
                    ", probability=" + probability +
                    ", bbox=" + Arrays.toString(bbox) +
                    '}';
        }
    }

    public static ArrayList<Detect> non_maximum_suppression(NDArray<Float, D2> array, float IoUThresh, float min_prob) {
        ArrayList<Integer> p = new ArrayList<>();

        NDArray<Float, D2> probs = ViewGettersAndSettersKt.slice(array, SliceKt.rangeTo(4, new SliceEndStub()), 1);
        NDArray<Integer, D1> idx = math.argMaxD2(probs, 1);

        int[] idx_array = new int[idx.getShape()[0]];
        Iterator<Integer> it = idx.iterator();
        for (int i = 0; i < idx_array.length; i++) {
            int val = it.next();
            idx_array[i] = val;
        }

        MultiArray<Float, D1> max_prob = ViewGettersAndSettersKt.view(probs, idx_array, new int[]{1}).reshape(idx_array.length);

        for (int i = 0; i < array.getShape()[0]; i++) {

        }

        ArrayList<Detect> results = new ArrayList<>();

        return results;
    }

    public static float[] xywh_to_xyxy(float x, float y, float w, float h) {
        float half_w = w/2.0f;
        float half_h = h/2.0f;
        return new float[]{x-half_w, y-half_h, x+half_w, y+half_h};
    }

    public static float[] xywh_to_xyxy(float[] box) {
        if (box.length != 4) throw new IllegalArgumentException("box array must have 4 values");
        return xywh_to_xyxy(box[0], box[1], box[2], box[3]);
    }

    public static float box_area(float x1, float y1, float x2, float y2) {
        return (x2 - x1) * (y2 - y1);
    }

    public static float box_area(float[] box) {
        if (box.length != 4) throw new IllegalArgumentException("box array must have 4 values");
        return box_area(box[0], box[1], box[2], box[3]);
    }

    public static float box_area_xywh(float x, float y, float w, float h) {
        return w*h;
    }

    public static float box_area_xywh(float[] box) {
        if (box.length != 4) throw new IllegalArgumentException("box array must have 4 values");
        return box_area_xywh(box[0], box[1], box[2], box[3]);
    }

    public static boolean has_positive_surface(float x1, float y1, float x2, float y2) {
        return (x2 >= x1) && (y2 >= y1);
    }
    public static boolean has_positive_surface(float[] box) {
        if (box.length != 4) throw new IllegalArgumentException("box array must have 4 values");
        return has_positive_surface(box[0], box[1], box[2], box[3]);
    }

    public static boolean has_positive_surface_xywh(float x, float y, float w, float h) {
        return (w >= 0.0) && (h >= 0.0);
    }
    public static boolean has_positive_surface_xywh(float[] box) {
        if (box.length != 4) throw new IllegalArgumentException("box array must have 4 values");
        return has_positive_surface_xywh(box[0], box[1], box[2], box[3]);
    }

    public static float iou(final float[] box1, final float[] box2) {
        float area1 = box_area(box1);
        float area2 = box_area(box2);
        float ix1 = Math.max(box1[0], box2[0]);
        float iy1 = Math.max(box1[1], box2[1]);
        float ix2 = Math.min(box1[2], box2[2]);
        float iy2 = Math.min(box1[3], box2[3]);
        float inter_area = box_area(ix1, iy1, ix2, iy2);
        if (!has_positive_surface(ix1, iy1, ix2, iy2)) {
            inter_area = 0.0f;
        }
        float union_area = area1 + area2 - inter_area;
        return inter_area / union_area;
    }

    public static ArrayList<Detect> non_maximum_suppression(float[] arr, int n_rows, double IoUThresh, double min_prob) {
        ArrayList<Pair<Integer, Float>> p = new ArrayList<>();

        int[] max_idx = new int[n_rows];
        Arrays.fill(max_idx, -1);
        float[] max_prob = new float[n_rows];
        Arrays.fill(max_prob, -1.0f);

        int n_cols = arr.length / n_rows;

        for (int i = 0; i < n_rows; i++) {
            float max_val = Float.MIN_VALUE;
            int idx_val = 0;
            for (int j = 4; j < n_cols; j++) {
                int cur_idx = n_cols * i + j;
                float val = arr[cur_idx];
                if (val > max_val) {
                    max_val = val;
                    idx_val = j;
                }
            }
            max_prob[i] = max_val;
            max_idx[i] = idx_val;
            if (max_val > min_prob) {
                float[] box = DetectUtils.xywh_to_xyxy(Arrays.copyOfRange(arr, i*n_cols, i*n_cols + 4));
                float box_area = box_area(box);
                boolean positive_surface = has_positive_surface(box);
                if (positive_surface && box_area > 0.0) p.add(new Pair<>(i, max_val));
            }
        }

        ArrayList<Detect> results = new ArrayList<>();

        //int[] sorted = ArrayUtils.argsort(max_prob, false);

        Collections.sort(p, (p1, p2) -> p2.second.compareTo(p1.second));

        for (int i = 0; i < p.size(); i++) {
            int idx1 = n_cols * p.get(i).first;
            float[] box1 = Arrays.copyOfRange(arr, idx1, idx1 + 4);
            box1 = xywh_to_xyxy(box1);
            for (int j = i+1; j < p.size(); j++) {
                int idx2 = n_cols * p.get(j).first;
                float[] box2 = Arrays.copyOfRange(arr, idx2, idx2 + 4);
                box2 = xywh_to_xyxy(box2);
                float iou = iou(box1, box2);
                if (iou > IoUThresh) {
                    p.remove(j);
                    --j;
                }
            }
        }

        for (int i = 0; i < p.size(); i++) {
            Pair<Integer, Float> pair = p.get(i);
            int idx = pair.first;
            int cls_idx = max_idx[idx] - 4; // remove 4 because of the 4 bounding box coordinates
            float prob = max_prob[idx];
            float[] box = Arrays.copyOfRange(arr, n_cols*idx, n_cols*idx + 4);
            Detect detect = new DetectUtils.Detect(cls_idx, prob, box);
            results.add(detect);
        }

        return results;
    }
}
