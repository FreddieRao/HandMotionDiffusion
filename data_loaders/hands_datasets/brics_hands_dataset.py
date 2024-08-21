import os
from os.path import join as pjoin
import numpy as np
import torch
from torch.utils import data
from data_loaders.hands_datasets.utils.word_vectorizer import WordVectorizer
from data_loaders.hands_datasets.utils.get_opt import get_opt
import json
import ujson
import random

global ROOT
ROOT = False

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
    keypoints3d_left = np.asarray(keypoints3d_left)[:, :, :3]
    keypoints3d_right = np.asarray(keypoints3d_right)[:, :, :3]
    gt_motion = np.concatenate((keypoints3d_left, keypoints3d_right), axis=1) # [F, 2*J, 3]
    return gt_motion

def transform_root_based_rep(array):
    root_slice = array[:, 0, :]
    modified_finger_slices = array[:, 1:, :]
    finger_slices = modified_finger_slices - root_slice[:, np.newaxis, :] 
    new_array = np.concatenate((root_slice[:, np.newaxis, :], finger_slices), axis=1)  # Shape [F, 21, 3]
    if ROOT:
        return new_array
    else:
        return array

def transform_space_based_rep(array):
    root_slice = array[:, 0, :]
    finger_slices = array[:, 1:, :]
    modified_finger_slices = finger_slices + root_slice[:, np.newaxis, :] 
    new_array = np.concatenate((root_slice[:, np.newaxis, :], modified_finger_slices), axis=1)  # Shape [F, 21, 3]
    if ROOT:
        return new_array
    else:
        return array
    
class TextOnlyDatasetVG(data.Dataset):
    def __init__(self, opt, mean, std):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 360
        self.pointer = 0
        # self.fixed_length = 120

        self.text_list = ['hands']

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, item):

        # Randomly select a caption
        text_data = random.choice(self.text_list)
        caption = text_data
        return None, None, caption, None, np.array([0]), self.fixed_length, None

'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetVG(data.Dataset):
    def __init__(self, opt, mean, std, w_vectorizer):
        self.opt = opt
        self.mean = mean
        self.std = std
        self.w_vectorizer = w_vectorizer
        self.max_length = 360
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 30
        self.rep = opt.rep

        data_list = []
        motion_dirs =  opt.motion_dirs
        text_dirs = opt.text_dirs

        if self.rep == 'keypoints':
            # TODO Text file loader
            
            train_data_info = {}
            all_scripts = []
            all_scripts_test = []
            for m_path, t_file in zip(motion_dirs, text_dirs):
                with open(t_file, 'r') as f:
                    script_lines = f.readlines()

                id_list = []
                for tid, (v_dir, script) in enumerate(zip(os.listdir(m_path),script_lines)):
                    v_path = os.path.join(m_path, v_dir)
                    if os.path.isdir(v_path):
                        chosen_frames = self.get_chose_frame(v_path)
                        if len(chosen_frames) >= 120 and len(chosen_frames) <200: # 4s-6.7s
                            if tid % 5 != 0:
                                data = {'path': v_path, 'caption': script.replace('\n', ''), 'chosen_frames': chosen_frames}
                                data_list.append(data)
                                all_scripts.append(script.replace('\n', ''))
                                train_data_info[script.replace('\n', '')] = (m_path.split('/')[-2], int(v_dir))
                            else:
                                all_scripts_test.append(script.replace('\n', ''))
                                train_data_info[script.replace('\n', '')] = (m_path.split('/')[-2], int(v_dir))
            save_dir = self.opt.args.save_dir
            text_save_dir = pjoin(save_dir, 'texts.txt')
            train_info_save_dir = pjoin(save_dir, 'train_info.json')
            with open(text_save_dir, 'w') as f:
                for script in all_scripts:
                    f.write(script + '\n')
                
                f.write('\n\n')

                for script in all_scripts_test:
                    f.write(script + '\n')
                                    
            with open(train_info_save_dir, 'w') as f:
                json.dump(train_data_info, f)
        else:
            # Manos loader
            raise ValueError(f'Unsupported data representation [{self.rep}]')

        self.data_list = data_list[:]  #SPECIAL!!!! For overfitting only
        print(len(self.data_list))

    def get_chose_frame(self, keypoints3d_dir):
        chosen_path_left = os.path.join(keypoints3d_dir, f"chosen_frames_left.json")
        chosen_path_right = os.path.join(keypoints3d_dir, f"chosen_frames_right.json")
        with open(chosen_path_left, "r") as f:
            chosen_frames_left = set(json.load(f))
        with open(chosen_path_right, "r") as f:
            chosen_frames_right = set(json.load(f))
        chosen_frames =list(set(chosen_frames_left | chosen_frames_right))
        return chosen_frames
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data_path = self.data_list[int(item)]['path']
        keypt3d_file_left = os.path.join(data_path, "left.jsonl")
        keypt3d_file_right = os.path.join(data_path, "right.jsonl")
        chosen_frames = self.data_list[int(item)]['chosen_frames']
        keypoints3d_left = []
        keypoints3d_right = []
        with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
            for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
                if l_idx in chosen_frames:
                    keypoints3d_left.append(np.array(ujson.loads(linel)))
                    keypoints3d_right.append(np.array(ujson.loads(liner)))
        keypoints3d_left = transform_root_based_rep(np.asarray(keypoints3d_left)[:, :, :3])
        keypoints3d_right = transform_root_based_rep(np.asarray(keypoints3d_right)[:, :, :3])
        motion = np.concatenate((keypoints3d_left, keypoints3d_right), axis=1) # [F, 2*J, 3]
        
        motion = (motion - self.mean) / self.std
        m_length = motion.shape[0]
        caption = self.data_list[int(item)]['caption']
        
        if m_length > self.max_motion_length:
            print('DownSampling')
            selected_indices = random.sample(range(m_length), self.max_motion_length)
            selected_indices.sort()
            motion = motion[selected_indices]
            m_length = self.max_motion_length
        
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)

        return None, None, caption, None, motion, m_length, None


# A wrapper class for t2m original dataset for MDM purposes
class BricsHands(data.Dataset):
    def __init__(self, mode, dataset_opt_path='./dataset/brics_hands_opt.txt', args=None, split="train", rep = 'keypoints', single_person=False, single_scene=False, **kwargs):
        self.mode = mode
        self.split = split
        self.rep = rep
        self.single_person = single_person
        self.single_scene = single_scene
        
        # self.dataset_name = 'brics_hands'
        # self.dataname = 'brics_hands'

        # Configurations of T2M dataset and KIT dataset is almost the same
        opt = get_opt(dataset_opt_path)
        if self.single_person and self.single_scene:
            # one scene - making tea
            dirs = ['2024-06-28-action-tea']
        elif self.single_person and not self.single_scene:
            # one person - rao 
            dirs = ['2024-06-28-action-noodle', '2024-06-28-action-salad-sandwich', '2024-06-28-action-tea', '2024-06-28-fastfood', '2024-06-30-action-baking', '2024-06-30-action-folder', '2024-06-30-action-gopro', '2024-06-30-action-laptop', '2024-06-30-action-mindmap', '2024-06-30-action-present', '2024-07-01-02-action-boxing', '2024-07-01-02-action-dog', '2024-07-01-02-action-instruments', '2024-07-01-02-action-monopoly', '2024-07-01-02-action-plant', '2024-07-01-02-action-stone', '2024-07-01-02-action-tablet', '2024-07-01-02-action-tool', '2024-07-01-03-action-cleaning', '2024-07-01-03-action-firstaid', '2024-07-01-03-action-makeup', '2024-07-01-03-action-massage', '2024-07-01-03-action-packing', '2024-07-01-03-action-sewing']
        
        opt.rep = rep
        if rep == 'keypoints':
            opt.motion_dirs = [pjoin(opt.data_root, d, 'keypoints_3d') for d in dirs]
            opt.joints_num = 42
            opt.dim_pose = 42 * 3
        elif rep == 'mano':
            opt.motion_dirs = [pjoin(opt.data_root, d, 'params') for d in dirs]
        else:
            raise ValueError(f'Unsupported data representation [{rep}]')
    
        opt.text_dirs = [pjoin(opt.data_root, d, 'scripts.txt') for d in dirs]
        opt.max_motion_length = 200
        opt.args = args
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        mean_path = 'dataset/tea_mean_root.npy' if ROOT else 'dataset/tea_mean_space.npy'
        std_path = 'dataset/tea_std_root.npy' if ROOT else 'dataset/tea_std_space.npy'
        
        if self.single_person and self.single_scene:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        else:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)         

        if self.mode == 'text_only':
            self.t2m_dataset = TextOnlyDatasetVG(self.opt, self.mean, self.std)
        else:
            # TODO Replace with new dictionary
            self.w_vectorizer = WordVectorizer('glove', 'our_vab')
            self.t2m_dataset = Text2MotionDatasetVG(self.opt, self.mean, self.std, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()