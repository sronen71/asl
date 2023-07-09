import cv2
import pandas as pd
import glob
import os
from protobuf_to_dict import protobuf_to_dict  # pip install protobuf3-to-dict
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import mediapipe as mp  # noqa: E402

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def visualize_landmarks(image, results):
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
        )
    cv2.imshow("MediaPipe Holistic", image)


def holistic(folder, visualize=False):
    frames = glob.glob(folder + "/*")
    frames = sorted(frames)
    i = 0
    sequence_results = []
    with mp_holistic.Holistic(
        model_complexity=2,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while i < len(frames):
            image = cv2.imread(frames[i])
            i += 1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            # face_landmarks=results.face_landmarks
            # pose_landmarks=results.pose_landmarks
            # left_hand_landmarks=results.left_hand_landmarks
            # right_hand_landmarks=results.right_hand_landmarks
            if visualize:
                visualize_landmarks(image, results)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            sequence_results.append(results)
        return sequence_results


def process_results(sequence_results, sequence_id):
    rows = []
    for i, results in enumerate(sequence_results):
        face_landmarks = results.face_landmarks
        left_hand_landmarks = results.left_hand_landmarks
        pose_landmarks = results.pose_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        parts = {
            "face": face_landmarks,
            "left_hand": left_hand_landmarks,
            "pose": pose_landmarks,
            "right_hand": right_hand_landmarks,
        }
        parts_one_row = []
        for part in parts:
            if parts[part] is not None:
                landmarks = protobuf_to_dict(parts[part])
            else:
                continue
            df = pd.DataFrame(landmarks["landmark"])
            if "visibility" in df.columns:
                df.drop(columns="visibility", inplace=True)
            columns = df.columns
            data = df.values.T.flatten()
            # print(part,len(data))
            df_part_one_row = pd.DataFrame(
                [data], columns=[f"{c}_{part}_{i}" for c in columns for i in range(len(df))]
            )
            parts_one_row.append(df_part_one_row)
        if len(parts_one_row) > 0:
            df_one_row = pd.concat(parts_one_row, axis=1)
            df_one_row["frame"] = i
            df_one_row["sequence_id"] = sequence_id
        else:
            df_one_row = pd.DataFrame({"frame": [i], "sequence_id": [sequence_id]})
        columns = df_one_row.columns
        new_columns = (
            ["sequence_id", "frame"]
            + [col for col in columns if "x_" in col]
            + [col for col in columns if "y_" in col]
            + [col for col in columns if "z_" in col]
        )
        df_one_row = df_one_row[new_columns]
        rows.append(df_one_row)
    df_seq = pd.concat(rows, axis=0)
    df_seq.set_index("sequence_id", inplace=True)
    # print(df_seq.shape)
    return df_seq


def main():
    path1 = "ChicagoFSWild"
    path2 = "ChicagoFSWildPlus"
    if "chicago.csv" in glob.glob("*.csv"):
        df_done = pd.read_csv("chicago.csv")
        num_done = len(df_done)
    else:
        df_done = None
        num_done = 0
    data1 = pd.read_csv(
        path1 + "/" + path1 + ".csv", converters={"signer": str}
    )  # filename,label_proc
    data1["path"] = path1
    data2 = pd.read_csv(
        path2 + "/" + path2 + ".csv", converters={"signer": str}
    )  # filename,label_proc
    data2["path"] = path2
    data = pd.concat([data1, data2], axis=0)
    data = data[~data["label_proc"].isna()]
    filenames = data["filename"].unique()
    random.seed(1)
    random.shuffle(filenames)
    chicago = []
    output_file_id_base = 9000000
    sequence_id_base = 9000000
    count = num_done
    seqs = []
    batch = 1000
    for file1 in filenames[num_done:]:
        drow = data[data["filename"] == file1]
        label = drow["label_proc"].item()
        signer = drow["signer"].item()
        path = drow["path"].item()
        print(count, path, file1, label)
        folder = path + "/Frames/" + file1
        sequence_id = sequence_id_base + count
        output_file_id = output_file_id_base + count // batch
        record = {}
        record["file_path"] = path + "/" + file1
        record["file_id"] = output_file_id
        record["sequence_id"] = sequence_id
        record["participant_id"] = signer
        record["phrase"] = label
        chicago.append(record)
        sequence_results = holistic(folder)
        df_seq1 = process_results(sequence_results, sequence_id)
        seqs.append(df_seq1)
        if (count + 1) % batch == 0:
            df = pd.concat(seqs, axis=0)
            df.to_parquet(f"input/chicago/{output_file_id}.parquet")
            seqs = []
            df_csv = pd.DataFrame(chicago)
            df_csv = pd.concat((df_done, df_csv), axis=0)
            df_csv.to_csv("chicago.csv", index=False)
        count += 1


if __name__ == "__main__":
    main()
