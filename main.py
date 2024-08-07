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

# Load the neural network model globally
net = cv.dnn.readNetFromTensorflow("model/graph_opt.pb")

def extract_points(image_path, thr=0.2, width=368, height=368):
    frame = cv.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read the image from {image_path}")

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
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

    return points, frame

def estimate_depth(side_points):
    # Use side image points to estimate shoulder depth
    if side_points[BODY_PARTS["Neck"]] and side_points[BODY_PARTS["RHip"]]:
        neck = side_points[BODY_PARTS["Neck"]]
        r_hip = side_points[BODY_PARTS["RHip"]]

        # Calculate the depth of the shoulder
        shoulder_depth_px = math.sqrt((neck[0] - r_hip[0]) ** 2 + (neck[1] - r_hip[1]) ** 2)
        return shoulder_depth_px
    else:
        raise ValueError("Necessary points for calculating shoulder depth not detected")

def estimate_pose(front_image_path, side_image_path, person_height_cm, thr=0.2, width=368, height=368, output_width=800, output_height=600):
    front_points, front_frame = extract_points(front_image_path, thr, width, height)
    side_points, side_frame = extract_points(side_image_path, thr, width, height)

    if front_points[BODY_PARTS["LShoulder"]] and front_points[BODY_PARTS["RShoulder"]] and front_points[BODY_PARTS["Neck"]] and front_points[BODY_PARTS["RHip"]] and front_points[BODY_PARTS["RKnee"]] and front_points[BODY_PARTS["RAnkle"]]:
        left_shoulder = front_points[BODY_PARTS["LShoulder"]]
        right_shoulder = front_points[BODY_PARTS["RShoulder"]]
        neck = front_points[BODY_PARTS["Neck"]]
        r_hip = front_points[BODY_PARTS["RHip"]]
        r_knee = front_points[BODY_PARTS["RKnee"]]
        r_ankle = front_points[BODY_PARTS["RAnkle"]]

        # Compute the pixel heights of different segments
        neck_to_ankle_px = math.sqrt((neck[0] - r_ankle[0]) ** 2 + (neck[1] - r_ankle[1]) ** 2)
        hip_to_knee_px = math.sqrt((r_hip[0] - r_knee[0]) ** 2 + (r_hip[1] - r_knee[1]) ** 2)
        knee_to_ankle_px = math.sqrt((r_knee[0] - r_ankle[0]) ** 2 + (r_knee[1] - r_ankle[1]) ** 2)
        
        # Calculate scaling factors
        total_height_px = neck_to_ankle_px + hip_to_knee_px + knee_to_ankle_px
        scaling_factor = person_height_cm / total_height_px

        # Calculate pixel distance between shoulders
        shoulder_length_px = math.sqrt((left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)

        # Calculate shoulder depth
        shoulder_depth_px = estimate_depth(side_points)

        # Calculate actual shoulder width in cm
        shoulder_width_cm = shoulder_length_px * scaling_factor
        shoulder_depth_cm = shoulder_depth_px * scaling_factor

        # Use the depth to adjust the shoulder width
        adjusted_shoulder_width_cm = math.sqrt((shoulder_width_cm / 2) ** 2 + shoulder_depth_cm ** 2) * 2

        print(f"Initial Shoulder Width (2D): {shoulder_width_cm:.2f} cm")
        print(f"Adjusted Shoulder Width: {shoulder_depth_cm:.2f} cm")
        
    else:
        print("Necessary points for calculating shoulder width not detected")

    # Visualize key points on the front image
    for i, point in enumerate(front_points):
        if point is not None:
            cv.circle(front_frame, point, 5, (0, 255, 0), -1)
            cv.putText(front_frame, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw pose pairs
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if front_points[idFrom] and front_points[idTo]:
            cv.line(front_frame, front_points[idFrom], front_points[idTo], (0, 255, 0), 3)
            cv.ellipse(front_frame, front_points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(front_frame, front_points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Calculate processing time
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(front_frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Resize the frame to the desired output size
    resized_frame = cv.resize(front_frame, (output_width, output_height))

    # Display the result
    cv.imshow('OpenPose using OpenCV', resized_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage:
# Provide the real-world height of the person in centimeters.
person_height_cm = 155.0  # Replace with the actual height of the person in cm

estimate_pose('files/6/front_img.jpg', 'files/6/side_img.jpg', person_height_cm)
