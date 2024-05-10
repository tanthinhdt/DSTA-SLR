import cv2
import random
import numpy as np
from mediapipe.python.solutions import pose


SELECTED_JOINTS = {
    27: {
        'pose': [0, 16, 14, 12, 15, 13, 11],
        'hand': [4, 8, 12, 16, 20, 1, 5, 9, 13, 17],
    },  # 27
}


def pad(joints: np.ndarray, num_frames: int = 150) -> np.ndarray:
    '''
    Add padding to the joints.

    Parameters
    ----------
    joints : np.ndarray
        The joints to pad.
    num_frames : int, default=150
        The number of frames to pad.

    Returns
    -------
    np.ndarray
        The padded joints.
    '''
    if joints.shape[0] < num_frames:
        L = joints.shape[0]
        padded_joints = np.zeros((num_frames, joints.shape[1], joints.shape[2]))
        padded_joints[:L, :, :] = joints
        rest = num_frames - L
        num = int(np.ceil(rest / L))
        pad = np.concatenate([joints for _ in range(num)], 0)[:rest]
        padded_joints[L:, :, :] = pad
    else:
        padded_joints = joints[:num_frames]
    return padded_joints


def extract_joints(
    source: str,
    keypoints_detector,
    resize_to: tuple = (256, 256),
    num_joints: int = 27,
    num_frames: int = 150,
    num_bodies: int = 1,
    num_channels: int = 3,
) -> np.ndarray:
    '''
    Extract the joints from the video.

    Parameters
    ----------
    source : str
        The path to the video.
    keypoints_detector : mediapipe.solutions.holistic.Holistic
        The keypoints detector.
    resize_to : tuple, default=(256, 256)
        The size to resize the image.
    num_joints : int, default=27
        The number of joints.
    num_frames : int, default=150
        The number of frames.
    num_bodies : int, default=1
        The number of bodies.
    num_channels : int, default=3
        The number of channels.

    Returns
    -------
    np.ndarray
        The extracted joints.
    '''
    cap = cv2.VideoCapture(source)

    extracted_joints = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.resize(image, resize_to)
        image = cv2.flip(image, flipCode=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_joints = []

        results = keypoints_detector.process(image)

        pose = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['pose'])
        if results.pose_landmarks is not None:
            pose = [
                (landmark.x * resize_to[0], landmark.y * resize_to[1], landmark.visibility)
                for i, landmark in enumerate(results.pose_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['pose']
            ]
        frame_joints.extend(pose)

        left_hand = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['hand'])
        if results.left_hand_landmarks is not None:
            left_hand = [
                (landmark.x * resize_to[0], landmark.y * resize_to[1], landmark.visibility)
                for i, landmark in enumerate(results.left_hand_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['hand']
            ]
        frame_joints.extend(left_hand)

        right_hand = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['hand'])
        if results.right_hand_landmarks is not None:
            right_hand = [
                (landmark.x * resize_to[0], landmark.y * resize_to[1], landmark.visibility)
                for i, landmark in enumerate(results.right_hand_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['hand']
            ]
        frame_joints.extend(right_hand)

        assert len(frame_joints) == num_joints, \
            f'Expected {num_joints} joints, got {len(frame_joints)} joints.'
        extracted_joints.append(frame_joints)

    extracted_joints = np.array(extracted_joints)
    extracted_joints = pad(extracted_joints, num_frames=num_frames)

    fp = np.zeros(
        (num_frames, num_joints, num_channels, num_bodies),
        dtype=np.float32,
    )
    fp[:, :, :, 0] = extracted_joints

    return np.transpose(fp, [2, 0, 1, 3])


def preprocess(
    source: str,
    keypoints_detector,
    normalization: bool = True,
    random_choose: bool = True,
    window_size: int = 120,
) -> np.ndarray:
    '''
    Preprocess the video.

    Parameters
    ----------
    source : str
        The path to the video.
    keypoints_detector : mediapipe.solutions.holistic.Holistic
        The keypoints detector.
    normalization : bool, default=True
        Whether to normalize the data.
    random_choose : bool, default=True
        Whether to randomly sample the data.
    window_size : int, default=120
        The window size.

    Returns
    -------
    np.ndarray
        The processed inputs for model.
    '''
    inputs = extract_joints(source=source, keypoints_detector=keypoints_detector)

    T = inputs.shape[1]
    ori_data = inputs
    for t in range(T - 1):
        inputs[:, t, :, :] = ori_data[:, t + 1, :, :] - ori_data[:, t, :, :]
    inputs[:, T - 1, :, :] = 0

    if random_choose:
        inputs = random_sample_np(inputs, window_size)
    else:
        inputs = uniform_sample_np(inputs, window_size)

    if normalization:
        assert inputs.shape[0] == 3
        inputs[0, :, :, :] = inputs[0, :, :, :] - inputs[0, :, 0, 0].mean(axis=0)
        inputs[1, :, :, :] = inputs[1, :, :, :] - inputs[1, :, 0, 0].mean(axis=0)

    return inputs[np.newaxis, :].astype(np.float32)


def random_sample_np(data: np.ndarray, size: int) -> np.ndarray:
    '''
    Sample the data randomly.

    Parameters
    ----------
    data : np.ndarray
        The data to sample.
    size : int
        The size of the data to sample.

    Returns
    -------
    np.ndarray
        The sampled data.
    '''
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data[:, random_list]


def uniform_sample_np(data: np.ndarray, size: int) -> np.ndarray:
    '''
    Sample the data uniformly.

    Parameters
    ----------
    data : np.ndarray
        The data to sample.
    size : int
        The size of the data to sample.

    Returns
    -------
    np.ndarray
        The sampled data.
    '''
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data[:, uniform_list]


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
