import cv2
import pandas as pd
import glob
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import mediapipe as mp # noqa: E402

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def visualize_landmarks(image,results):
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    for hand_landmarks in [results.left_hand_landmarks,results.right_hand_landmarks]:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )
    cv2.imshow('MediaPipe Holistic', image)


def holistic(folder):
    frames=glob.glob(folder+"/*")
    frames=sorted(frames)
    i=0
    with mp_holistic.Holistic(
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while (i<len(frames)):
            image = cv2.imread(frames[i])
            i+=1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            face_landmarks=results.face_landmarks
            pose_landmarks=results.pose_landmarks
            left_hand_landmarks=results.left_hand_landmarks
            right_hand_landmarks=results.right_hand_landmarks
            landmark_array = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]) #Nx3
            visualize_landmarks(image,results)
            if cv2.waitKey(5) & 0xFF == 27:
                break

def main():
    path="ChicagoFSWild"
    data=pd.read_csv(path+"/"+path+".csv") #filename,label_proc
    filenames=data["filename"].unique()
    file1=filenames[0]
    label=data[data["filename"]==file1]["label_proc"].item()
    print(file1,label)
    folder=path+"/Frames/"+file1
    holistic(folder)
    


if __name__=="__main__":
    main()

