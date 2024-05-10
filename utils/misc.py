import numpy as np
from mediapipe.python.solutions import pose


def calculate_angle(
    shoulder: list,
    elbow: list,
    wrist: list,
) -> float:
    '''
    Calculate the angle between the shoulder, elbow, and wrist.

    Parameters
    ----------
    shoulder : list
        Shoulder coordinates.
    elbow : list
        Elbow coordinates.
    wrist : list
        Wrist coordinates.

    Returns
    -------
    float
        Angle in degree between the shoulder, elbow, and wrist.
    '''
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)

    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) \
        - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def do_hands_relax(
    pose_landmarks: list,
    angle_threshold: float = 160.0,
) -> bool:
    '''
    Check if the hand is down.

    Parameters
    ----------
    hand_landmarks : list
        Hand landmarks.
    angle_threshold : float, optional
        Angle threshold, by default 160.0.

    Returns
    -------
    bool
        True if the hand is down, False otherwise.
    '''
    if pose_landmarks is None:
        return True

    landmarks = pose_landmarks.landmark
    left_shoulder = [
        landmarks[pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[pose.PoseLandmark.LEFT_SHOULDER.value].y,
        landmarks[pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
    ]
    left_elbow = [
        landmarks[pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[pose.PoseLandmark.LEFT_ELBOW.value].y,
        landmarks[pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
    ]
    left_wrist = [
        landmarks[pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[pose.PoseLandmark.LEFT_WRIST.value].y,
        landmarks[pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
    ]
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    right_shoulder = [
        landmarks[pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        landmarks[pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
    ]
    right_elbow = [
        landmarks[pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[pose.PoseLandmark.RIGHT_ELBOW.value].y,
        landmarks[pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
    ]
    right_wrist = [
        landmarks[pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[pose.PoseLandmark.RIGHT_WRIST.value].y,
        landmarks[pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
    ]
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    is_visible = all(
        [
            left_shoulder[2] > 0,
            left_elbow[2] > 0,
            left_wrist[2] > 0,
            right_shoulder[2] > 0,
            right_elbow[2] > 0,
            right_wrist[2] > 0,
        ]
    )

    return all(
        [
            is_visible,
            left_angle < angle_threshold,
            right_angle < angle_threshold,
        ]
    )
