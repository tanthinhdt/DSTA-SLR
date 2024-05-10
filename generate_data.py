import os
import cv2
import pickle
import logging
import argparse
import numpy as np
from mediapipe.python.solutions import holistic


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


SELECTED_JOINTS = {
    27: {
        'pose': [0, 16, 14, 12, 15, 13, 11],
        'hand': [4, 8, 12, 16, 20, 1, 5, 9, 13, 17],
    },  # 27
}
MAX_FRAME = 150
NUM_JOINTS = 27
NUM_CHANNELS = 3
MAX_BODY_TRUE = 1


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate data for the model.')
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='Split to generate data for.',
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='Directory containing the videos.',
    )
    parser.add_argument(
        '--label-path',
        type=str,
        required=True,
        help='Path to the labels.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the generated data.',
    )
    parser.add_argument(
        '--overwrite-data-joint',
        action='store_true',
        help='Overwrite the existing data.',
    )
    parser.add_argument(
        '--overwrite-npy',
        action='store_true',
        help='Overwrite the existing data.',
    )
    return parser.parse_args()


def extract_joints(
    source: str,
    keypoints_detector,
    num_joints: int = 27,
    new_size: tuple = (256, 256),
) -> np.ndarray:
    cap = cv2.VideoCapture(source)

    extracted_joints = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.resize(image, new_size)
        image = cv2.flip(image, flipCode=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_joints = []

        results = keypoints_detector.process(image)

        pose = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['pose'])
        if results.pose_landmarks is not None:
            pose = [
                (landmark.x * new_size[0], landmark.y * new_size[1], landmark.visibility)
                for i, landmark in enumerate(results.pose_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['pose']
            ]
        frame_joints.extend(pose)

        left_hand = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['hand'])
        if results.left_hand_landmarks is not None:
            left_hand = [
                (landmark.x * new_size[0], landmark.y * new_size[1], landmark.visibility)
                for i, landmark in enumerate(results.left_hand_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['hand']
            ]
        frame_joints.extend(left_hand)

        right_hand = [(0.0, 0.0, 0.0)] * len(SELECTED_JOINTS[num_joints]['hand'])
        if results.right_hand_landmarks is not None:
            right_hand = [
                (landmark.x * new_size[0], landmark.y * new_size[1], landmark.visibility)
                for i, landmark in enumerate(results.right_hand_landmarks.landmark)
                if i in SELECTED_JOINTS[num_joints]['hand']
            ]
        frame_joints.extend(right_hand)

        assert len(frame_joints) == num_joints, \
            f'Expected {num_joints} joints, got {len(frame_joints)} joints.'
        extracted_joints.append(frame_joints)

    return np.array(extracted_joints)


def main(args: argparse.Namespace) -> None:
    logging.info('Initializing...')
    label_dict = dict()
    with open(args.label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            label_dict[line[0]] = int(line[1])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'npy'), exist_ok=True)
    fp = np.zeros(
        (len(label_dict), MAX_FRAME, NUM_JOINTS, NUM_CHANNELS, MAX_BODY_TRUE),
        dtype=np.float32
    )

    keypoints_detector = holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True,
    )

    data_joint_path = os.path.join(args.output_dir, f'{args.split}_data_joint.npy')
    if args.overwrite_data_joint or not os.path.exists(data_joint_path):
        logging.info(f'Extracting joints for {args.split} split...')
        logging.info(f'Number of videos: {len(label_dict)}')
        for i, file_name in enumerate(label_dict.keys()):
            logging.info(f'Processing video {i + 1}/{len(label_dict)}: {file_name}')
            video_path = os.path.join(args.video_dir, file_name + '.mp4')

            npy_path = os.path.join(args.output_dir, 'npy', f'{file_name}.npy')
            if args.overwrite_npy or not os.path.exists(npy_path):
                skel = extract_joints(
                    source=video_path,
                    keypoints_detector=keypoints_detector,
                    num_joints=NUM_JOINTS,
                )
                np.save(npy_path, skel)
                logging.info('Extracted joints')
            else:
                skel = np.load(npy_path)
                logging.info('Loaded joints')

            if skel.shape[0] < MAX_FRAME:
                L = skel.shape[0]
                fp[i, :L, :, :, 0] = skel
                rest = MAX_FRAME - L
                pad = np.concatenate(
                    [skel for _ in range(int(np.ceil(rest / L)))], axis=0
                )[:rest]
                fp[i, L:, :, :, 0] = pad
            else:
                fp[i, :, :, :, 0] = skel[:MAX_FRAME, :, :]
        fp = np.transpose(fp, [0, 3, 1, 2, 4])
        logging.info(f'Data shape: {fp.shape}')
        np.save(data_joint_path, fp)

    with open(os.path.join(args.output_dir, f'{args.split}_label.pkl'), 'wb') as f:
        pickle.dump((list(label_dict.keys()), list(label_dict.values())), f)
    logging.info('Data saved successfully.')
    logging.info('Done!')


if __name__ == '__main__':
    args = get_args()
    main(args=args)
