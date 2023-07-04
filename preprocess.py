import json
import glob
import pandas as pd
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402


ROWS_PER_FRAME = 543
input_path = "input/"


def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_char_dict():
    char_dict_file = f"{input_path}/character_to_prediction_index.json"
    with open(char_dict_file) as f:
        char_dict = json.load(f)

    char_dict["SOS"] = 59
    char_dict["EOS"] = 60
    char_dict["PAD"] = -1
    return char_dict


def main():
    output_path = "output/"
    char_dict = get_char_dict()
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
    for file_name in files:
        print(file_name)
        fold += 1
        file_id = file_name.split("/")[-1].split(".")[0]
        df = pd.read_parquet(file_name)
        labels = dtrain[dtrain["file_id"].astype(str) == file_id]
        unique_seqs = df.index.unique()
        output_file = f"fold_{fold%5}_" + file_name.split("/")[-1].replace("parquet", "tfrecord")
        if file_name in files1:
            folder = "records/"
        else:
            folder = "sup_records/"
        output_file = output_path + folder + output_file

        with tf.io.TFRecordWriter(output_file, options=options) as writer:
            for seq in unique_seqs:
                phrase = labels[labels["sequence_id"] == seq]["phrase"].item()
                label = [char_dict[x] for x in phrase]
                frames = df.loc[seq]
                # print(file_id, seq, phrase)
                if frames.ndim < 2:
                    continue
                frames_numpy = frames.iloc[:, 1:].to_numpy()
                frames_numpy = (
                    frames_numpy.reshape(-1, 3, ROWS_PER_FRAME)
                    .transpose([0, 2, 1])  # Now it's (None,ROW_PER_FRAME,3)
                    .flatten()
                    .astype(np.float32)
                )
                # f1 = frames.reshape(-1, ROWS_PER_FRAME, 3)
                features_dict = {
                    "coordinates": _float_array_feature(frames_numpy),
                    "label": _int_array_feature(label),
                    "sequence_id": _int_array_feature([seq]),
                }
                features = tf.train.Features(feature=features_dict)
                example_proto = tf.train.Example(features=features)
                example = example_proto.SerializeToString()
                writer.write(example)


if __name__ == "__main__":
    main()
