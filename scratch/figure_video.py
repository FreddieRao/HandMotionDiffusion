import imageio
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from PIL import Image

import os
import cv2
import sys
import json
import ujson

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from data_loaders.hands_datasets.utils.easymocap_utils import vis_smpl_stack, projectN3, vis_repro, load_model
from data_loaders.hands_datasets.utils.video_handler import create_video_writer, convert_video_ffmpeg
import data_loaders.hands_datasets.utils.camera_utils as param_utils
from data_loaders.hands_datasets.brics_hands_dataset import transform_space_based_rep, load_3d_keypoints
sys.path.append("../pose_estimation/third-party/EasyMocap")
from easymocap.dataset import CONFIG
from easymocap.pipeline import smpl_from_keypoints3d
from easymocap.smplmodel import select_nf
os.environ["PYOPENGL_PLATFORM"] = "egl"

def read_params(params_path):
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")

    return params

def get_projections(params, cam_names, n_Frames):
    # Gets the projection matrices and distortion parameters
    projs = []
    intrs = []
    dist_intrs = []
    dists = []
    rot = []
    trans = []

    for param in params:
        if (param["cam_name"] == cam_names):
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)

            rot.append(r)
            trans.append(t)

            intrs.append(intr.copy())
            
            dist_intrs.append(intr.copy())

            projs.append(intr @ extr)
            dists.append(dist)

    cameras = { 
        'K': np.repeat(np.asarray(intrs), n_Frames, axis=0),
        'R': np.repeat(np.asarray(rot), n_Frames, axis=0), 
        'T': np.repeat(np.asarray(trans), n_Frames, axis=0),
        'dist': np.repeat(np.asarray(dists), n_Frames, axis=0),
        'P': np.repeat(np.asarray(projs), n_Frames, axis=0)
        }

    return intrs, np.asarray(projs), dist_intrs, dists, cameras


def stack_images(image_list):
    # Convert all RGB images to RGBA (full opacity)
    for i in range(len(image_list)):
        if image_list[i].shape[2] == 3:  # If image is RGB
            image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
            image_list[i] = np.dstack([image_list[i], np.full((image_list[i].shape[0], image_list[i].shape[1]), 255, dtype=np.uint8)])  # Add alpha channel

    # Start with the first image
    base_image = image_list[0].copy()

    # Overlay all subsequent images on top
    for img in image_list[1:]:
        # Convert to PIL Image for easy blending
        base_image_pil = Image.fromarray(base_image)
        overlay_image_pil = Image.fromarray(img)
        
        # Composite the images
        base_image_pil = Image.alpha_composite(base_image_pil, overlay_image_pil)
        
        # Convert back to numpy array
        base_image = np.array(base_image_pil)

    return base_image

# -------------------- Visualization Functions -------------------- #
def plot_3d_hand_motion(gt_motion, src_root_path=None, out_path=None, save_file=None, to_vis_smpl=True):

    parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--text_prompt', type=str)
    parser.add_argument('--input_text', type=str)
    parser.add_argument('--num_repetitions', type=int)
    parser.add_argument('--model', type=str, default='manor')
    parser.add_argument('--body', type=str, default='handr')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--robust3d', type=bool, default=False)
    parser.add_argument('--save_origin', type=bool, default=False)
    parser.add_argument('--save_frame', type=bool, default=False)
    
    args = parser.parse_args()
    
    # Total Frames
    n_Frames = gt_motion.shape[0]
    # Loads the camera parameters
    params_path = os.path.join(src_root_path, "optim_params.txt")
    params = read_params(params_path)

    # Load image for fitting
    firstframe_name = [_ for _ in os.listdir(src_root_path) if _.startswith("visframes")][0]
    firstframe_view = firstframe_name[firstframe_name.find('_') + 1:]
    firstframe_path = os.path.join(src_root_path, firstframe_name, 'vis.png')
    intrs, projs, dist_intrs, dists, cameras = get_projections(params, firstframe_view, n_Frames)
    image = cv2.imread(firstframe_path)
    image = cv2.undistort(image, intrs[0], dists[0], None)

    body_model_right = load_model(gender='neutral', model_type='manor', model_path="body_models", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)
    body_model_left = load_model(gender='neutral', model_type='manol', model_path="body_models", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)

    # fits the mano model
    dataset_config = CONFIG['handr']
    weight_pose = {
        'k3d': 1e2, 'k2d': 2e-3,
        'reg_poses': 1e-3, 'smooth_body': 1e2, 'smooth_poses': 1e2,
    }

    gt_n_Frames = gt_motion.shape[0]
    gt_motion = np.pad(gt_motion, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    gt_motion_right = gt_motion[:,21:,:]
    gt_motion_left = gt_motion[:,:21,:]
    gt_params_right = smpl_from_keypoints3d(body_model_right, gt_motion_right, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
    gt_params_left = smpl_from_keypoints3d(body_model_left, gt_motion_left, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)        
            
    # visualize model
    import trimesh
    
    images = [image]# * n_Frames
    
    nf = 0
    
    image_list = [image]

    for abs_idx in tqdm(range(n_Frames), total=n_Frames):
        a_min = 30
        a_max = 60
        inter = 2
        if abs_idx % inter == 0 and abs_idx<=a_max and abs_idx>a_min:
            inter_factor = (abs_idx - a_min) / (a_max - a_min)
            vertices = []
            faces = []
            colors = []

            gt_param_right = select_nf(gt_params_right, nf)
            gt_param_left = select_nf(gt_params_left, nf)
            gt_vertices_right = body_model_right(return_verts=True, return_tensor=False, **gt_param_right)
            gt_vertices_left = body_model_left(return_verts=True, return_tensor=False, **gt_param_left)
            gt_vertice = np.concatenate((gt_vertices_left[0], gt_vertices_right[0]), axis=0)
            gt_face = np.concatenate((body_model_left.faces, body_model_right.faces+gt_vertices_left[0].shape[0]), axis=0)
            vertices.append(gt_vertice)
            faces.append(gt_face)                               

            image_vis = vis_smpl_stack(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, inter_factor=inter_factor)
            image_list.append(image_vis[0])
                        
        nf += 1
    return image_list


folder = '2024-06-28-action-salad-sandwich'
motion_id = 23
motion_dir = os.path.join('/users/rfu7/ssrinath/datasets/Action/brics-mini', folder)
gt_motion_path = os.path.join(motion_dir, 'keypoints_3d', str(motion_id).zfill(3))
gt_motion = load_3d_keypoints(gt_motion_path)
image_list = plot_3d_hand_motion(gt_motion, src_root_path=motion_dir, out_path='./scratch', save_file='haha.png', to_vis_smpl=True)
result_image = stack_images(image_list)

# Convert the result to an image and save it
result_image_pil = Image.fromarray(result_image)
result_image_pil.save('scratch/haha_salad.png')