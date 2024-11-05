package com.example.androidassistant.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.regex.Pattern;

public class StringUtils {

    /**
     * Split a string using a separator but keep the separator in the string parts.
     * Ex: str="abcdefg" sep="c"
     * return ["ab","cdefg"]
     * @param str
     * @param sep
     * @return
     */
    public static String[] splitBefore(String str, String sep) {
        String[] parts = str.split(Pattern.quote(sep));

        for (int i = 1; i < parts.length; i++) {
            parts[i] = sep + parts[i];
        }

        return parts;
    }

    /**
     * Split a string using a separator but keep the separator in the string parts.
     * Ex: str="abcdefg" sep="c"
     * return ["abc","defg"]
     * @param str
     * @param sep
     * @return
     */
    public static String[] splitAfter(String str, String sep) {
        String[] parts = str.split(Pattern.quote(sep));

        for (int i = 0; i < parts.length-1; i++) {
            parts[i] = parts[i] + sep;
        }

        return parts;
    }


    public static ArrayList<String> splitAround(String str, String sep) {
        String[] parts = str.split(Pattern.quote(sep));

        ArrayList<String> result = new ArrayList<>(parts.length);
        if (parts.length > 1) {
            for (int i = 0; i < parts.length - 1; i++) {
                result.add(parts[i]);
                result.add(sep);
            }
        }
        result.add(parts[parts.length-1]);

        return result;
    }


    public static ArrayList<String> splitAroundAny(String str, Collection<String> seps) {
        ArrayList<String> result = new ArrayList<>(1);
        result.add(str);

        for (String sep : seps) {
            for (int i = 0; i < result.size();) {
                String current = result.get(i);
                ArrayList<String> splits = splitAround(current, sep);
                result.remove(i);
                for (int j = 0; j < splits.size(); j++) {
                    String splitPart = splits.get(j);
                    result.add(i+j, splitPart);
                }
                i += splits.size();
            }
        }
        return result;
    }

}
