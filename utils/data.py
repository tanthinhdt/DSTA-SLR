import cv2
import random
import numpy as np


SELECTED_JOINTS = {
    27: {
        'pose': [0, 16, 14, 12, 15, 13, 11],
        'hand': [4, 8, 12, 16, 20, 1, 5, 9, 13, 17],
    },  # 27
}


def pad(joints: np.ndarray, num_frames: int = 150) -> np.ndarray:
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

    Returns
    -------
    dict
        The model inputs.
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
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data[:, random_list]


def uniform_sample_np(data: np.ndarray, size: int) -> np.ndarray:
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data[:, uniform_list]
