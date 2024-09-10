import os
import numpy as np
import json
import ujson
from tqdm import tqdm

def load_3d_keypoints(path):
    keypt3d_file_left = os.path.join(path, "left.jsonl")
    keypt3d_file_right = os.path.join(path, "right.jsonl")
    chosen_path_left = os.path.join(path, f"chosen_frames_left.json")
    chosen_path_right = os.path.join(path, f"chosen_frames_right.json")
    with open(chosen_path_left, "r") as f:
        chosen_frames_left = set(json.load(f))
    with open(chosen_path_right, "r") as f:
        chosen_frames_right = set(json.load(f))
    chosen_frames =list(set(chosen_frames_left | chosen_frames_right))
    keypoints3d_left = []
    keypoints3d_right = []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                keypoints3d_left.append(np.array(ujson.loads(linel)))
                keypoints3d_right.append(np.array(ujson.loads(liner)))
    if len(keypoints3d_left) == 0 or len(keypoints3d_right) == 0:
        return None
    keypoints3d_left = np.asarray(keypoints3d_left)[:, :, :3]
    keypoints3d_right = np.asarray(keypoints3d_right)[:, :, :3]
    gt_motion = np.concatenate((keypoints3d_left, keypoints3d_right), axis=1) # [F, 2*J, 3]
    return gt_motion

main_path = '/users/rfu7/ssrinath/datasets/Action/brics-mini'

VALID_SCENES_RAO = ['2024-06-28-action-noodle', '2024-06-28-action-salad-sandwich', '2024-06-28-action-tea', '2024-06-28-fastfood', '2024-06-30-action-baking', '2024-06-30-action-folder', '2024-06-30-action-gopro', '2024-06-30-action-laptop', '2024-06-30-action-mindmap', '2024-06-30-action-present', '2024-07-01-02-action-boxing', '2024-07-01-02-action-dog', '2024-07-01-02-action-instruments', '2024-07-01-02-action-monopoly', '2024-07-01-02-action-plant', '2024-07-01-02-action-stone', '2024-07-01-02-action-tablet', '2024-07-01-02-action-tool', '2024-07-01-03-action-cleaning', '2024-07-01-03-action-firstaid', '2024-07-01-03-action-makeup', '2024-07-01-03-action-massage', '2024-07-01-03-action-packing', '2024-07-01-03-action-sewing']
VALID_SCENES_ALEXJ = ['2024-06-09','2024-06-12','2024-06-17','2024-06-18','2024-06-20', '2024-07-22-action-alexj-file', '2024-07-22-action-alexj-plant-mindmap', '2024-07-28-action-alexj-gopro', '2024-07-28-action-alexj-laptop', '2024-07-28-action-alexj-sewing', '2024-06-28-action-noodle']
VALID_SCENES_JASON = ['2024-08-14-action-jason-instruments', '2024-08-14-action-jason-monopoly', '2024-08-14-action-jason-puppy', '2024-08-14-action-json-boxing']
VALID_SCENES_ANGELA = ['2024-08-15-action-angela-plant', '2024-08-15-action-angela-sewing', '2024-08-15-action-angela-stone', '2024-08-15-action-angela-toolbox']
VALID_SCENES_JULIA = ['2024-07-18-action-julia-packing', '2024-07-18-tion-julia-age', '2024-07-19-action-julia-cleaning', '2024-07-19-action-julia-firstaid'] # '2024-07-18-action-julia-makeup', 
VALID_SCENES_KYLIE = ['2024-07-19-action-kylie-instruments', '2024-07-19-action-kylie-makeup', '2024-07-19-action-kylie-massage', '2024-07-19-action-kylie-sewing']
VALID_SCENES_ALEXW = ['2024-07-20-action-alexw-book', '2024-07-20-action-alexw-file', '2024-07-20-action-alexw-present', '2024-07-20-action-alexw-gopro'] 
VALID_SCENES_NIRAYKA = ['2024-07-23-action-nirayka-boxing', '2024-07-23-action-nirayka-dog', '2024-07-23-action-nirayka-instruments', '2024-07-23-action-nirayka-monopoly', '2024-07-23-action-nirayka-toolbox']
VALID_SCENES_SUDARSHAN = ['2024-07-24-action-sudarshan-book', '2024-07-24-action-sudarshan-file', '2024-07-24-action-sudarshan-laptop', '2024-07-24-action-sudarshan-present']
VALID_SCENES_PRACCHO = ['2024-07-25', '2024-07-25-action-praccho-massage', '2024-07-25-action-praccho-plant', '2024-07-25-action-praccho-stone', '2024-07-25-action-praccho-tool']
VALID_SCENES_CHAERIN = ['2024-07-10-action-fastfood', '2024-07-10-action-noodles', '2024-07-10-action-sandwich', '2024-07-10-action-tea', '2024-08-12-action-chaerin-baking', '2024-08-12-action-chaerin-salad']
VALID_SCENES_ARMAN = ['2024-07-29-action-arman-tea', '2024-07-30-action-arman-fastfood', '2024-07-30-action-arman-noodles']
VALID_SCENES_SRINATH = ['2024-07-30-action-srinath-sandwich', '2024-07-31-action-srinath-toolbench']
VALID_SCENES_ALEXG = ['2024-07-31-action-alexg-file', '2024-07-31-action-alexg-mindmap']
VALID_SCENES_RAHUL = ['2024-07-31-action-rahul-baking', '2024-07-31-action-rahul-monopoly']
VALID_SCENES_ALL = VALID_SCENES_RAO + VALID_SCENES_JASON + VALID_SCENES_ANGELA \
    + VALID_SCENES_JULIA + VALID_SCENES_KYLIE + VALID_SCENES_ALEXW + VALID_SCENES_NIRAYKA + VALID_SCENES_SUDARSHAN + VALID_SCENES_PRACCHO + VALID_SCENES_CHAERIN \
    + VALID_SCENES_ARMAN +  VALID_SCENES_SRINATH + VALID_SCENES_ALEXG + VALID_SCENES_RAHUL

all_motion_array = []
for scene in tqdm(VALID_SCENES_ALL[:]):
    scene_path = os.path.join(main_path, scene, 'keypoints_3d')
    for seq in os.listdir(scene_path):
        path = os.path.join(scene_path, seq)
        if 'right.jsonl' in os.listdir(path):
            motion = load_3d_keypoints(path)
            if motion is not None:
                all_motion_array.append(motion)

all_motion = np.concatenate(all_motion_array, axis=0)
mean = np.mean(all_motion, axis=0, keepdims=True)
std = np.std(all_motion, axis=0, keepdims=True)
normalized_data = (all_motion - mean) / std
print(mean)
print('-'*20)
print(std)
print(normalized_data.min(), normalized_data.max())
np.save('dataset/hand_mean_0731.npy', mean)
np.save('dataset/hand_std_0731.npy', std)