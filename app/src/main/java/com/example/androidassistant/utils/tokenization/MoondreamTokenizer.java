package com.example.androidassistant.utils.tokenization;


import android.util.Pair;

import com.example.androidassistant.utils.StringUtils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.regex.Pattern;

public class MoondreamTokenizer {
    public static String WHITESPACE_CHAR = "Ġ";
    public static String NEWLINE_CHAR = "Ċ";

    private HashMap<String, Integer> token_to_id_map;
    private HashMap<Integer, String> id_to_token_map;
    private ArrayList<Pair<String, String>> merges; // should not use android Pair as they do not implement hashCode() apparently
    private HashMap<Pair<Integer, Integer>, Integer> id_merges;

    private ArrayList<String> special_tokens;

    private int bos_token;
    private int eos_token;
    private int unk_token;

    public MoondreamTokenizer(List<Pair<String, Integer>> vocabulary, List<Pair<String, String>> merges) {
        token_to_id_map = new HashMap<>();
        id_to_token_map = new HashMap<>();

        for (Pair<String, Integer> pair : vocabulary) {
            token_to_id_map.put(pair.first, pair.second);
            id_to_token_map.put(pair.second, pair.first);
        }

        this.merges = new ArrayList<>();
        this.merges.addAll(merges);

         this.merges.remove(new Pair<>(NEWLINE_CHAR, NEWLINE_CHAR));

        this.id_merges = new HashMap<>();
        for (Pair<String, String> pair : merges) {
            id_merges.put(new Pair<>(token_to_id(pair.first), token_to_id(pair.second)),
                    token_to_id(pair.first + pair.second));
        }

        // remove the newlines merge from the merges since the python tokenizer does not do this
        // id_merges.remove(new Pair<>(NEWLINE_CHAR, NEWLINE_CHAR)); // does not work because Pair is not hashable

        String eos_token_str = "<|endoftext|>";

        special_tokens = new ArrayList<>();
        special_tokens.add(eos_token_str);

        this.eos_token = this.token_to_id(eos_token_str);
        this.bos_token = eos_token;
        this.unk_token = eos_token;
    }

    public static MoondreamTokenizer from_files(String vocab_path, String merges_path) {
        File vocab_file = new File(vocab_path);
        File merges_file = new File(merges_path);

        List<Pair<String, Integer>> vocabulary = new ArrayList<>();
        List<Pair<String, String>> merges = new ArrayList<>();

        // read vocab file

        JSONObject json_vocab = null;

        try {
            StringBuilder sb = new StringBuilder();
            Files.readAllLines(vocab_file.toPath(), Charset.defaultCharset()).forEach(sb::append);
            json_vocab = new JSONObject(sb.toString());
        } catch(IOException | JSONException e)
        {
            throw new RuntimeException(e);
        }

        final JSONObject final_json_vocab = json_vocab;

        final_json_vocab.keys().forEachRemaining(key -> {
            try {
                vocabulary.add(new Pair<>(key, final_json_vocab.getInt(key)));
            } catch (JSONException e) {
                throw new RuntimeException(e);
            }
        });


        // read merges
        try {
            Files.readAllLines(merges_file.toPath(), Charset.defaultCharset())
                    .stream().skip(1).forEach(line -> {
                String[] parts = line.split(" ");
                merges.add(new Pair<>(parts[0], parts[1]));
            });
        } catch(IOException e)
        {
            throw new RuntimeException(e);
        }

        return new MoondreamTokenizer(vocabulary, merges);
    }

    public int get_eos_token() {
        return this.eos_token;
    }

    public int get_bos_token() {
        return this.bos_token;
    }

    public int get_unk_token() {
        return this.unk_token;
    }

    public String token_id_to_string(int id) {
        return this.id_to_token_map.get(id);
    }

    public int token_to_id(String token) {
        return this.token_to_id_map.get(token);
    }

    public String preprocess(String str) {
        return str.replace(" ", WHITESPACE_CHAR).replace("\n", NEWLINE_CHAR);
    }

    public String postprocess(String str) {
        return str.replace(WHITESPACE_CHAR, " ").replace(NEWLINE_CHAR, "\n");
    }

    public ArrayList<Integer> encore_non_special(String text) {
        ArrayList<Integer> ids = new ArrayList<>();

        String[] parts = StringUtils.splitBefore(text, " ");

        for (String part: parts) {
            String preprocessed = preprocess(part);
            ArrayList<Integer> subIds = new ArrayList<>(preprocessed.length());
            for (char c : preprocessed.toCharArray()) {
                int tokenId = token_to_id(String.valueOf(c));
                subIds.add(tokenId);
            }
            ids.addAll(subIds);
        }

        for (Pair<String, String> merge : merges) {
            int i = 0;
            while (i < ids.size() - 1) {
                int t1 = ids.get(i);
                int t2 = ids.get(i+1);

                String t1s = token_id_to_string(t1);
                String t2s = token_id_to_string(t2);

                if (t1s.equals(merge.first) && t2s.equals(merge.second)) {
                    ids.set(i, token_to_id(merge.first + merge.second));
                    ids.remove(i+1);
                }
                else {
                    i++;
                }
            }

        }

        return ids;
    }

    public ArrayList<Integer> encode(String text) {
        ArrayList<Object> partially_processed = new ArrayList<>();

        partially_processed.add(text);

        for (String special_token : special_tokens) {
            for (int i = 0; i < partially_processed.size();) {
                Object x = partially_processed.get(i);

                if (x instanceof String) {
                    String string = (String) x;
                    String[] parts = string.split(Pattern.quote(special_token),-1);

                    if (parts.length > 1) {
                        partially_processed.remove(i);
                        for (int j=0; j < parts.length - 1; j++) {
                            partially_processed.add(parts[j]);
                            partially_processed.add(token_to_id(special_token));
                        }
                        partially_processed.add(parts[parts.length - 1]);
                    }
                    i += parts.length * 2 - 1;
                }
                else {
                    i++;
                }
            }

        }

        ArrayList<Integer> ids = new ArrayList<>();

        for (Object ob: partially_processed) {
            if (ob instanceof String) {
                String string = (String) ob;
                if (!string.isEmpty()){
                    ArrayList<Integer> regular_ids = encore_non_special(string);
                    ids.addAll(regular_ids);
                }
            } else if (ob instanceof Integer) {
                int id = (int) ob;
                ids.add(id);
            }
            else {
                throw new RuntimeException("object should be either an integer or a String");
            }
        }

        return ids;
    }


    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(id_to_token_map.get(id));
        }
        return postprocess(sb.toString());
    }

    public String decode(int id) {
        ArrayList<Integer> list = new ArrayList<>(1);
        list.add(id);
        return decode(list);
    }

}
