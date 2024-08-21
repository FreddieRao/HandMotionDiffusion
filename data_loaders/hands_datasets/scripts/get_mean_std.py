import os
import json
import ujson
import numpy as np

def transform_root_based_rep(array):
    root_slice = array[:, 0, :]
    finger_slices = array[:, 1:, :]
    modified_finger_slices = finger_slices - root_slice[:, np.newaxis, :] 
    new_array = np.concatenate((root_slice[:, np.newaxis, :], modified_finger_slices), axis=1)  # Shape [F, 21, 3]
    return array
    
main_path = '/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-06-28-action-tea/keypoints_3d'

all_left_array = []
all_right_array = []
for vid in os.listdir(main_path):
    v_path = os.path.join(main_path, vid)
    keypt3d_file_left = os.path.join(v_path, "left.jsonl")
    keypt3d_file_right = os.path.join(v_path, "right.jsonl")
    
    chosen_path_left = os.path.join(v_path, f"chosen_frames_left.json")
    chosen_path_right = os.path.join(v_path, f"chosen_frames_right.json")
    with open(chosen_path_right, "r") as f:
        chosen_frames_right = set(json.load(f))
    with open(chosen_path_left, "r") as f:
        chosen_frames_left = set(json.load(f))
    chosen_frames =list(set(chosen_frames_right | chosen_frames_left))
    
    keypoints3d_left = []
    keypoints3d_right = []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                keypoints3d_left.append(np.array(ujson.loads(linel)))
                keypoints3d_right.append(np.array(ujson.loads(liner)))

    if len(np.asarray(keypoints3d_left).shape) > 1:
        keypoints3d_left = transform_root_based_rep(np.asarray(keypoints3d_left)[:, :, :3]) #.reshape(-1, 21 * 3)
        all_left_array.append(keypoints3d_left)
    if len(np.asarray(keypoints3d_right).shape) > 1:
        keypoints3d_right = transform_root_based_rep(np.asarray(keypoints3d_right)[:, :, :3]) #.reshape(-1,  21 * 3)
        all_right_array.append(keypoints3d_right)

all_left = np.concatenate(all_left_array, axis=0)
all_right = np.concatenate(all_right_array, axis=0)

# Calculate mean along the F * J dimension (axis 0)
mean_array1 = np.mean(all_left, axis=0, keepdims=True)
mean_array2 = np.mean(all_right, axis=0, keepdims=True)
all_mean = np.concatenate((mean_array1, mean_array2), axis=1)
# Calculate variance along the F * J dimension (axis 0)
variance_array1 = np.std(all_left, axis=0, keepdims=True)
variance_array2 = np.std(all_right, axis=0, keepdims=True)
all_var = np.concatenate((variance_array1, variance_array2), axis=1)

np.save('dataset/tea_mean_space.npy', all_mean)
np.save('dataset/tea_std_space.npy', all_var)
