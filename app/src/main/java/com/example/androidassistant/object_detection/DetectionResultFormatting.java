package com.example.androidassistant.object_detection;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class DetectionResultFormatting {

    private static float middle_left_boundary = 0.3f;

    private static int MAX_DETECTIONS_BY_GROUP = 5;


    public static String formatDistance(float distance) {
        if (distance < 1.0f) {
            return "moins de 1 mètre";
        }
        else if (distance > 15.0f) {
            return "plus de 15 mètres";
        }

        int meters = Math.round(distance);
        return meters + " mètres";
    }

    public static String formatList(List<String> list) {
        if (list.isEmpty()) {
            return null;
        }
        else if (list.size() == 1) {
            return list.get(0);
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < list.size() - 1; i++) {
            String sep = (i == list.size() - 2) ? " et " : ", ";
            sb.append(list.get(i)).append(sep);
        }

        sb.append(list.get(list.size() - 1));

        return sb.toString();
    }

    public static String formatDetectionResult(List<DetectionWithDepth> detections) {
        if (detections.isEmpty()) {
            return "Je ne vois aucun objet.";
        }

        ArrayList<DetectionWithDepth> middle = new ArrayList<>(detections);

        // split into three groups based on x position

        ArrayList<DetectionWithDepth> left = new ArrayList<>();
        ArrayList<DetectionWithDepth> right = new ArrayList<>();

        assert middle_left_boundary < 0.5;

        for (int i = 0; i < middle.size(); i++) {
            DetectionWithDepth d = middle.get(i);
            if (d.x < middle_left_boundary) {
                left.add(d);
                middle.remove(i);
                i--;
            } else if (d.x > 1 - middle_left_boundary) {
                right.add(d);
                middle.remove(i);
                i--;
            }
        }

        String left_result = formatGroupResult(left);
        String middle_result = formatGroupResult(middle);
        String right_result = formatGroupResult(right);

        StringBuilder sb = new StringBuilder();

        if (left_result != null) {
            sb.append("Sur la gauche il y a ").append(left_result).append(". ");
        }

        if (middle_result != null) {
            sb.append("Au milieu il y a ").append(middle_result).append(". ");
        }

        if (right_result != null) {
            sb.append("Sur la droite il y a ").append(right_result).append(". ");
        }

        return sb.toString();
    }

    public static String formatGroupResult(ArrayList<DetectionWithDepth> group) {
        if (group.isEmpty()) {
            return null;
        }

        // sort by confidence from highest to lowest so that the less confident detections are at the end
        group.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        // limit the number of detected objects to avoid very long utterances prioritizing the most confident ones
        group = group.stream()
                .limit(MAX_DETECTIONS_BY_GROUP)
                .collect(Collectors.toCollection(ArrayList::new));

        // aggregate by label
        Map<String, List<DetectionWithDepth>> grouped = group.stream().collect(Collectors.groupingBy((d) -> d.object_label));

        StringBuilder sb = new StringBuilder();

        ArrayList<Map.Entry<String, List<DetectionWithDepth>>> sorted = new ArrayList<>(grouped.entrySet());
        for (int i = 0; i < sorted.size(); i++) {
            Map.Entry<String, List<DetectionWithDepth>> entry = sorted.get(i);
            String label = entry.getKey();
            List<DetectionWithDepth> detections = entry.getValue();

            // sort by depth so that we talk about the closest objects first
            detections.sort((a, b) -> Float.compare(a.depth, b.depth));

            if (detections.size() == 1) {
                String toAdd = label + " à " + formatDistance(detections.get(0).depth);
                sb.append(toAdd);
            }
            else {
                String toAdd = detections.size() + " " + label + " à ";
                sb.append(toAdd);

                ArrayList<String> distances = new ArrayList<>();
                for (DetectionWithDepth d : detections) {
                    distances.add(formatDistance(d.depth));
                }
                String list = formatList(distances);
                sb.append(list);
            }

            if (i < sorted.size() - 1) {
                sb.append(", ");
            }

        }

        String result = sb.toString();

        return result;
    }

}
