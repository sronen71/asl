import glob
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from .constants import Constants
from .config import CFG
from .utils import selected_columns


def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


input_path = CFG.input_path
output_path = CFG.output_path


def preprocess():
    files1 = glob.glob(input_path + "train_landmarks/*.parquet")
    files2 = glob.glob(input_path + "supplemental_landmarks/*.parquet")
    files = files1 + files2

    dtrain1 = pd.read_csv(input_path + "train.csv")
    dtrain2 = pd.read_csv(input_path + "supplemental_metadata.csv")
    dtrain = pd.concat([dtrain1, dtrain2])
    # print(dtrain[["file_id", "sequence_id", "participant_id"]].sort_values(by=["participant_id"]))
    # MAX_STRING_LEN = 43

    os.makedirs(output_path + "records/", exist_ok=True)
    fold = 0
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    columns = selected_columns(files[0])
    for file_name in files:
        print(file_name)
        fold += 1
        file_id = file_name.split("/")[-1].split(".")[0]
        df = pd.read_parquet(file_name, columns=columns)
        labels = dtrain[dtrain["file_id"].astype(str) == file_id]
        unique_seqs = df.index.unique()

        output_file = file_name.split("/")[-1].replace("parquet", "tfrecord")
        if "supp" in file_name:
            output_file = "supp_" + output_file
        output_file = output_path + "records/" + output_file

        with tf.io.TFRecordWriter(output_file, options=options) as writer:
            for seq in unique_seqs:
                phrase = labels[labels["sequence_id"] == seq]["phrase"].item()
                label = [Constants.char_dict[x] for x in phrase]
                frames = df.loc[seq]
                # print(file_id, seq, phrase)
                if frames.empty:
                    continue
                frames_numpy = frames.to_numpy().flatten().astype(np.float32)
                features_dict = {
                    "coordinates": _float_array_feature(frames_numpy),
                    "label": _int_array_feature(label),
                }
                features = tf.train.Features(feature=features_dict)
                example_proto = tf.train.Example(features=features)
                example = example_proto.SerializeToString()
                writer.write(example)
