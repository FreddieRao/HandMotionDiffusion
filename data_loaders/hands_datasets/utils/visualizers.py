import imageio
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import os
import cv2
import sys
import json
import ujson

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from data_loaders.hands_datasets.utils.easymocap_utils import vis_smpl, projectN3, vis_repro, load_model
from data_loaders.hands_datasets.utils.video_handler import create_video_writer, convert_video_ffmpeg, add_text_to_frame
import data_loaders.hands_datasets.utils.camera_utils as param_utils
from data_loaders.hands_datasets.brics_hands_dataset import transform_space_based_rep
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

# -------------------- Visualization Functions -------------------- #
def plot_3d_hand_motion(motion, gt_motion, src_root_path=None, ith=0, out_path=None, save_file=None, to_vis_smpl=True, save_mesh=False, vis_3d_repro=True):

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
    n_Frames = motion.shape[0]
    # Loads the camera parameters
    params_path = os.path.join(src_root_path, "optim_params.txt")
    params = read_params(params_path)

    # Load image for fitting
    firstframe_name = [_ for _ in os.listdir(src_root_path) if _.startswith("firstframes")][0]
    firstframe_view = firstframe_name[firstframe_name.find('_') + 1:]
    firstframe_path = os.path.join(src_root_path, firstframe_name, str(ith).zfill(3) + '.png')
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

    cof_motion_right = np.pad(transform_space_based_rep(motion[:,21:,:]), ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    cof_motion_left = np.pad(transform_space_based_rep(motion[:,:21,:]), ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    params_right = smpl_from_keypoints3d(body_model_right, cof_motion_right, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
    params_left = smpl_from_keypoints3d(body_model_left, cof_motion_left, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
    if gt_motion is not None:
        gt_n_Frames = gt_motion.shape[0]
        gt_motion = np.pad(gt_motion, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
        gt_motion_right = gt_motion[:,21:,:]
        gt_motion_left = gt_motion[:,:21,:]
        gt_params_right = smpl_from_keypoints3d(body_model_right, gt_motion_right, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
        gt_params_left = smpl_from_keypoints3d(body_model_left, gt_motion_left, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)        
            
    # visualize model
    if to_vis_smpl or save_mesh or vis_3d_repro:
        import trimesh
        
        images = [image]# * n_Frames
        text = save_file.replace('.mp4', '').replace('_', ' ')
        if to_vis_smpl:
            os.makedirs(f'{out_path}/mano', exist_ok=True)
            outhand_mano_path = f'{out_path}/mano/{save_file}'

        if vis_3d_repro:
            os.makedirs(f'{out_path}/repro_3d', exist_ok=True)
            outhand_3d_path = f'{out_path}/repro_3d/{save_file}'
        
        nf = 0
        
        for abs_idx in tqdm(range(n_Frames), total=n_Frames):
            ocean_breeze_scheme = np.array([[0, 255, 255], [0, 0, 139]])
            if to_vis_smpl:
                vertices = []
                faces = []
                colors = [ocean_breeze_scheme[0]/255]
                
                param_right = select_nf(params_right, nf)
                param_left = select_nf(params_left, nf)
                vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)
                vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)   
                vertice = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                face = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                vertices.append(vertice)
                faces.append(face)
                image_vis = vis_smpl(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, colors=colors, add_back=False, out_dir=outhand_mano_path)

                if gt_motion is not None and nf < gt_n_Frames:
                    vertices = []
                    faces = []
                    colors = [ocean_breeze_scheme[1]/255]
                    gt_param_right = select_nf(gt_params_right, nf)
                    gt_param_left = select_nf(gt_params_left, nf)
                    gt_vertices_right = body_model_right(return_verts=True, return_tensor=False, **gt_param_right)
                    gt_vertices_left = body_model_left(return_verts=True, return_tensor=False, **gt_param_left)
                    gt_vertice = np.concatenate((gt_vertices_left[0], gt_vertices_right[0]), axis=0)
                    gt_face = np.concatenate((body_model_left.faces, body_model_right.faces+gt_vertices_left[0].shape[0]), axis=0)
                    vertices.append(gt_vertice)
                    faces.append(gt_face) 
                    # TODO: Add ground. And find correct first frame.                         
                    image_vis_gt = vis_smpl(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, colors=colors, add_back=True, out_dir=outhand_mano_path)
                    image_vis = np.hstack((image_vis[:, :, :3], image_vis_gt[:, :, :3]))
                    image_vis = add_text_to_frame(image_vis, text)
                
                if nf == 0:
                    # TODO: Add text. 
                    outhand_mano = create_video_writer(outhand_mano_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)

                outhand_mano.write(image_vis)
                        
            vis_config = CONFIG['handlr']
            if vis_3d_repro:
                kpts_repros = []
                keypoints = np.concatenate((cof_motion_left[abs_idx], cof_motion_right[abs_idx]), axis=0)
                kpts_repro = projectN3(keypoints, projs)
                kpts_repro[:, :, 2] = 0.5
                kpts_repros.append(kpts_repro)
                if gt_motion is not None and nf < gt_n_Frames:
                    gt_kpts_repro = projectN3(gt_motion[abs_idx], projs)
                    kpts_repros.append(gt_kpts_repro)

                image_vis = vis_repro(args, images, kpts_repros, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path)
                if abs_idx == 0:
                    outhand_3d = create_video_writer(outhand_3d_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)
                outhand_3d.write(image_vis)

            # save the mesh
            if save_mesh:
                vertices = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                outdir = os.path.join(out_path, f'meshes/{str(ith).zfill(3)}')
                os.makedirs(outdir, exist_ok=True)
                outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                mesh.export(outname)
                
            nf += 1

        if to_vis_smpl:
            outhand_mano.release()
            print(outhand_mano_path)
            convert_video_ffmpeg(outhand_mano_path)
            print('Video Handler Released')
        if vis_3d_repro:
            outhand_3d.release()
            convert_video_ffmpeg(outhand_3d_path)
            print('Video Handler Released')


