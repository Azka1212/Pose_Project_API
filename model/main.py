import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


net=cv.dnn.readNetFromTensorflow("graph_opt.pb")


inWidth=368
inHeight=368
thr=0.2


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def Estimation(PATH, frame_skip=1):
    cap = cv.VideoCapture(PATH)
    cap.set(cv.CAP_PROP_FPS, 10)
    cap.set(3, 960)
    cap.set(4, 540)

    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError(0)
    
    all_keypoints = []  # List to store keypoints for each frame

    frame_counter = 0  # Counter to track frames

    while cap.isOpened():  # Loop until the video ends
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        
        frame_counter += 1

        # Process every nth frame
        if frame_counter % frame_skip != 0:
            continue
            
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > thr else None)

        all_keypoints.append(points)  # Store keypoints for this frame

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.8fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('OpenPose using OpenCV', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break
    
    cv.destroyAllWindows()  # Close OpenCV windows
    
    # Replace None values with (0, 0)
    for frame_keypoints in all_keypoints:
        for i in range(len(frame_keypoints)):
            if frame_keypoints[i] is None:
    #             print(frame_keypoints[i])
                frame_keypoints[i] = (0, 0)
    
    return all_keypoints  # Return list of keypoints for all frames

import cv2 as cv

# Constants
inWidth=368
inHeight=368
thr=0.2 # Confidence threshold

# Define the body parts
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# Define the pairs to connect keypoints
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]



def WebcamEstimation():
    cap = cv.VideoCapture(0)  # Use the default webcam (index 0)
    cap.set(cv.CAP_PROP_FPS, 10)
    cap.set(3, 800)
    cap.set(4, 800)

    if not cap.isOpened():
        raise IOError(0)
        
    all_keypoints = []  # List to store keypoints for each frame
    start_time = cv.getTickCount() / cv.getTickFrequency()

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        
        current_time = cv.getTickCount() / cv.getTickFrequency()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 25:  # Capture video for 10 seconds
            break
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > thr else None)

        all_keypoints.append(points)  # Store keypoints for this frame

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.8fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('OpenPose using OpenCV', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break
    
    cap.release()  # Release the webcam
    cv.destroyAllWindows()  # Close OpenCV windows
    
    # Replace None values with (0, 0)
    for frame_keypoints in all_keypoints:
        for i in range(len(frame_keypoints)):
            if frame_keypoints[i] is None:
                frame_keypoints[i] = (0, 0)
    
    return all_keypoints  # Return list of keypoints for all frames


def create_bounding_box(keypoints):
    x_coords = [point[0] for point in keypoints if point != (0, 0)]
    y_coords = [point[1] for point in keypoints if point != (0, 0)]

    if x_coords and y_coords:
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        return (min_x, min_y, max_x, max_y)
    else:
        return None

def normalize_keypoints(keypoints):
    # Normalize and reshape the keypoints array
    keypoints_array = np.array(keypoints)
    normalized_keypoints = keypoints_array / np.linalg.norm(keypoints_array)
    return normalized_keypoints

# Compare poses function (as previously defined)
def compare_poses(keypoints1, keypoints2):
    similarity = cosine_similarity(keypoints1, keypoints2)
    return similarity


def calculate_similarity(keypoints1, keypoints2):
    keypoints1_array = np.array(keypoints1)
    keypoints2_array = np.array(keypoints2)
    
    # Reshape keypoints arrays for cosine similarity
    keypoints1_array = keypoints1_array.reshape(1, -1)
    keypoints2_array = keypoints2_array.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(keypoints1_array, keypoints2_array)
    
    return similarity[0][0]  # Return the similarity value


# Compare skeletons function
def compare_skeletons(video1_keypoints, video2_keypoints):
    min_length = min(len(video1_keypoints), len(video2_keypoints))

    total_similarity = 0.0
    for keypoints1, keypoints2 in zip(video1_keypoints[:min_length], video2_keypoints[:min_length]):
        keypoints1 = [point for point in keypoints1 if point != (0, 0)]
        keypoints2 = [point for point in keypoints2 if point != (0, 0)]
        
        # Check if both keypoints arrays have the same number of keypoints
        if len(keypoints1) != len(keypoints2):
            continue  # Skip this frame
        
        keypoints1 = normalize_keypoints(keypoints1)
        keypoints2 = normalize_keypoints(keypoints2)
        
        similarity_score = calculate_similarity(keypoints1, keypoints2)
        total_similarity += similarity_score

    # Calculate the average similarity score
    avg_similarity = total_similarity / min_length

    # Compare the average similarity score to the threshold
    if avg_similarity > 0.07:
        print("Exercises match")
    else:
        print("Exercises do not match")
        
    print("Average Similarity Score:", avg_similarity)




# Example usage

# Give video paths for videos for comparing

video_path1 = 'video16.mp4'
video_path2 = ''

# Main Exercise Video
video1_keypoints = Estimation(video_path1)

# Comparing Video
# video2_keypoints = Estimation(video_path2)

# Webcam Video
video2_keypoints = WebcamEstimation()

compare_skeletons(video1_keypoints, video2_keypoints)