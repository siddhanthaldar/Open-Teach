import h5py as h5
import numpy as np
from pandas import read_csv
import pickle as pkl
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from sentence_transformers import SentenceTransformer

# FOLDER_NAMES = ["2024.04.23", "2024.04.24", "2024.04.25", "2024.04.26", "2024.04.29",
#                 "2024.04.30", "2024.05.01", "2024.05.05", "2024.05.06"]
FOLDER_NAMES = {
    # "2024.05.29": ["set_up_table"],
    # "2024.05.31": ["put_fruits_in_basket"],
    "2024.06.08": None,
}
for FOLDER_NAME, task_names in FOLDER_NAMES.items():
    PROCESSED_DATA_PATH = Path(f"/mnt/robotlab/siddhant/projects/scaling_polytask/processed_data/{FOLDER_NAME}")
    SAVE_DATA_PATH = Path(f"/mnt/robotlab/siddhant/projects/scaling_polytask/processed_data_pkl_aa_may8/")
    # task_names = None #["lift_pan_lid", "place_pan_lid"]
    camera_indices = [1,2,3,4,51,52]
    img_size = (128, 128)
    NUM_DEMOS = None

    # if task_names is None, get all task names
    if task_names is None:
        task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

    # Create the save path
    SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for TASK_NAME in task_names:
        DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

        if (SAVE_DATA_PATH / f'{TASK_NAME}.pkl').exists():
            print(f"Data for {TASK_NAME} already exists. Appending to it...")
            data = pkl.load(open(SAVE_DATA_PATH / f'{TASK_NAME}.pkl', "rb"))
            observations = data['observations']
            max_cartesian = data['max_cartesian']
            min_cartesian = data['min_cartesian']
            # max_joint_states = data['max_joint_states']
            # min_joint_states = data['min_joint_states']
            max_gripper = data['max_gripper']
            min_gripper = data['min_gripper']
            task_emb = data['task_emb']
        else:
            # Get task name sentence
            label_path = DATASET_PATH / "label.txt"
            task_name = label_path.read_text().strip()
            print(f"Task name: {task_name}")
            lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            task_emb = lang_model.encode(task_name)

            # Init storing variables
            observations = []

            # Store max and min
            max_cartesian, min_cartesian = None, None
            # max_joint_states, min_joint_states = None, None
            # max_rel_cartesian, min_rel_cartesian = None, None
            max_gripper, min_gripper = None, None

        # Load each data point and save in a list
        dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
        for i, data_point in enumerate(dirs):
            print(f"Processing data point {i+1}/{len(dirs)}")
            
            if NUM_DEMOS is not None:
                if int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
                    print(f"Skipping data point {data_point}")
                    continue

            observation = {}
            # images
            image_dir = data_point / "videos"
            if not image_dir.exists():
                print(f"Data point {data_point} is incomplete")
                continue
            for save_idx, idx in enumerate(camera_indices):
                # Read the frames in the video
                video_path = image_dir / f"camera{idx}.mp4"
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"Video {video_path} could not be opened")
                    continue
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx == 52:
                        # crop the right side of the image for the gripper cam
                        shape = frame.shape
                        crop_percent = 0.2
                        frame = frame[:, :int(shape[1] * (1 - crop_percent))]
                    frame = cv2.resize(frame, img_size)
                    frames.append(frame)
                observation[f"pixels{save_idx}"] = np.array(frames)
            # read cartesian and gripper states from csv
            csv_path = data_point / 'states.csv'
            state = read_csv(csv_path)
            # Read cartesian state where every element is a 6D pose
            # Separate the pose into values instead of string
            cartesian_states = state['pose_aa'].values
            cartesian_states = np.array([np.array([float(x.strip()) for x in pose[1:-1].split(',')]) for pose in cartesian_states], dtype=np.float32)
            # joint_states = state['joint_states'].values
            # for i in range(1, len(joint_states)):
            #     if joint_states[i] == '[]':
            #         joint_states[i] = joint_states[i-1]
            # joint_states = np.array([np.array([float(x.strip()) for x in joint_states[i][1:-1].split(',')]) for i in range(len(joint_states))], dtype=np.float32)
            # for pose in cartesian_states:
            #     for x in pose[1:-1].split(','):
            #         print(x.strip())
            #         x = float(x.strip())
            # # # Convert roll-pitch-yaw to sin-cos
            # # def wrap_angle(angles):
            # #     """
            # #     Ensure angle stays within [0, 2*pi) degrees range.
            # #     """
            # #     for idx in range(len(angles)):
            # #         for i in range(3):
            # #             while angles[idx][i] >= 2*np.pi:
            # #                 angles[idx][i] -= 2*np.pi
            # #             while angles[idx][i] < 0:
            # #                 angles[idx][i] += 2*np.pi
            # #     return angles
            # cartesian_pos = cartesian_states[:, :3]
            # cartesian_ori = cartesian_states[:, 3:]
            # cartesian_ori = Rotation.from_rotvec(cartesian_ori).as_euler('xyz')
            # # cartesian_ori = wrap_angle(cartesian_ori)
            # cartesian_ori = np.concatenate([np.sin(cartesian_ori), np.cos(cartesian_ori)], axis=1)
            # cartesian_states = np.concatenate([cartesian_pos, cartesian_ori], axis=1)
            # rest
            gripper_states = state['gripper_state'].values.astype(np.float32)
            observation["cartesian_states"] = cartesian_states.astype(np.float32)
            # observation["joint_states"] = joint_states.astype(np.float32)
            observation["gripper_states"] = gripper_states.astype(np.float32)
            
            # update max and min
            if max_cartesian is None:
                max_cartesian = np.max(cartesian_states, axis=0)
                min_cartesian = np.min(cartesian_states, axis=0)
            else:
                max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
                min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
            # if max_joint_states is None:
            #     max_joint_states = np.max(joint_states, axis=0)
            #     min_joint_states = np.min(joint_states, axis=0)
            # else:
            #     max_joint_states = np.maximum(max_joint_states, np.max(joint_states, axis=0))
            #     min_joint_states = np.minimum(min_joint_states, np.min(joint_states, axis=0))
            if max_gripper is None:
                max_gripper = np.max(gripper_states)
                min_gripper = np.min(gripper_states)
            else:
                max_gripper = np.maximum(max_gripper, np.max(gripper_states))
                min_gripper = np.minimum(min_gripper, np.min(gripper_states))
            
            # append to observations
            observations.append(observation)

        # Save the data
        data = {
            'observations': observations,
            'max_cartesian': max_cartesian,
            'min_cartesian': min_cartesian,
            # 'max_joint_states': max_joint_states,
            # 'min_joint_states': min_joint_states,
            'max_gripper': max_gripper,
            'min_gripper': min_gripper,
            'task_emb': task_emb
        }
        pkl.dump(data, open(SAVE_DATA_PATH / f'{TASK_NAME}.pkl', "wb"))