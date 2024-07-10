import cv2 as cv
import numpy as np
import math

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

def estimate_pose(image_path, person_height_cm, thr=0.2, width=368, height=368, output_width=800, output_height=600):
    inWidth = width
    inHeight = height

    net = cv.dnn.readNetFromTensorflow("human-pose-estimation-opencv-master/graph_opt.pb")

    frame = cv.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read the image from {image_path}")

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    # Visualize points for debugging
    for i, point in enumerate(points):
        if point is not None:
            cv.circle(frame, point, 5, (0, 255, 0), -1)
            cv.putText(frame, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

    # Calculate shoulder length
    if points[BODY_PARTS["LShoulder"]] and points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["Nose"]] and points[BODY_PARTS["RAnkle"]]:
        left_shoulder = points[BODY_PARTS["LShoulder"]]
        right_shoulder = points[BODY_PARTS["RShoulder"]]
        nose = points[BODY_PARTS["Nose"]]
        right_ankle = points[BODY_PARTS["RAnkle"]]
        
        # Compute the pixel height of the person from Nose to Right Ankle
        person_height_px = math.sqrt((nose[0] - right_ankle[0]) ** 2 + (nose[1] - right_ankle[1]) ** 2)
        
        # Calculate scaling factor
        scaling_factor = person_height_cm / person_height_px
        
        # Calculate pixel distance between shoulders
        shoulder_length_px = math.sqrt((left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)
        
        # Calculate actual shoulder length in cm
        shoulder_length_cm = shoulder_length_px * scaling_factor
        print(f"Shoulder length: {shoulder_length_cm:.2f} cm")
    else:
        print("Necessary points for calculating shoulder length not detected")

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Resize the frame to the desired output size
    resized_frame = cv.resize(frame, (output_width, output_height))

    cv.imshow('OpenPose using OpenCV', resized_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage:
# Provide the real-world height of the person in centimeters.
person_height_cm = 165.0  # Replace with the actual height of the person in cm

estimate_pose('D:/Fiverr Projects/Pose Project/files/test1.jpg', person_height_cm)
