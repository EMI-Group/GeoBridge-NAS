#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_select
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import json
# from utils.opengs_utlis import mask_feature_mean, get_SAM_mask_and_feat, load_code_book

np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)

from eval.openclip_encoder import OpenCLIPNetwork

import glob
from pathlib import Path



def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, scene_name, method_name, normalize_flag=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)




    mask_root = Path(model_path) / method_name
    target_text = sorted(path.stem for path in mask_root.glob("*.npy"))
    if not target_text:
        raise FileNotFoundError(f"No grounding masks found in {mask_root}")
    print("Grounded queries:", target_text)

    for t_i, query_name in enumerate(target_text):
        # if target_text[t_i] != "old camera":
        #     continue

        mask_file = f"{model_path}/{method_name}/{query_name}.npy"
        print("mask_file")
        if not os.path.exists(mask_file):
            continue

        Gaussian_mask = torch.from_numpy(np.load(mask_file)).cuda()

        empty_gaussian_mask = not bool(Gaussian_mask.reshape(-1).any())
        if empty_gaussian_mask:
            print(f"WARNING: empty Gaussian mask for {query_name}; rendering blank silhouettes")



        # 

        # render
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # note: evaluation frame
            scene_gt_frames = {
                "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
                "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
                "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
                "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"],
                "demo_data_recon": ['image_2','image_3','image_1','image_4']
            }
            candidate_frames = scene_gt_frames[scene_name]
            
            if  view.image_name not in candidate_frames:
                continue

            # print("normalize_flag", normalize_flag)
            # exit()
            if empty_gaussian_mask:
                rendered_cluster_imgs = background[:, None, None].expand(
                    3, view.image_height, view.image_width
                )
                rendered_leaf_cluster_silhouettes = torch.zeros(
                    (1, view.image_height, view.image_width, 1),
                    dtype=background.dtype,
                    device=background.device,
                )
            else:
                render_pkg = render_select(view, gaussians, pipeline, background, Gaussian_mask=Gaussian_mask, feature_mode=False, render_mode="RGB", feature_normalize=normalize_flag)

                #
                rendered_cluster_imgs = render_pkg["render"]
                print(rendered_cluster_imgs.shape)
                #
                rendered_leaf_cluster_silhouettes = render_pkg["mask"]
                print(rendered_leaf_cluster_silhouettes.shape)
            # exit()

            # 

            render_cluster_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster")
            render_cluster_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster_silhouette")
            makedirs(render_cluster_path, exist_ok=True)
            makedirs(render_cluster_silhouette_path, exist_ok=True)


            # 
                
                # print("hello???")
            torchvision.utils.save_image(rendered_cluster_imgs[:3,:,:], os.path.join(render_cluster_path, \
                view.image_name + f"_{query_name}.png"))
            # save object mask
            cluster_silhouette = rendered_leaf_cluster_silhouettes[0] > 0.7
            # print(cluster_silhouette.shape)
            cluster_silhouette = cluster_silhouette.cpu().permute(2, 0, 1)
            torchvision.utils.save_image(cluster_silhouette.to(torch.float32), os.path.join(render_cluster_silhouette_path, \
                view.image_name + f"_{query_name}.png"))

        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_mode: bool, feature_npy: bool, render_mode: str, method_name, normalize_flag): ###
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # 从.ply文件中加载场景信息

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)

        scene_name = dataset.source_path.split('/')[-1]
        print("scene_name:", scene_name)
        #
        
        checkpoint = f"{dataset.model_path}/chkpnt30000.pth"

        
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_v1(model_params)




        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dataset.speedup = getattr(dataset, 'speedup', False)

        # def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_name):

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, f"text2obj_prompt_{method_name}", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene_name, method_name, normalize_flag)

        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--feature_mode', action='store_true', help='use feature replace RGB')
    parser.add_argument("--feature_npy", action='store_true', help='store 16-dim feature map in npy')
    parser.add_argument("--render_mode", default="RGB", type=str) # RGB+ED
    parser.add_argument("--method_name", default="RGB", type=str) # RGB+ED
    parser.add_argument("--normalize_flag", action="store_true") # RGB+ED
    args = get_combined_args(parser) #从命令行获取参数，并根据得到的model_path解析该目录下cfg_args文件，获取训练参数
    print("Rendering " + args.model_path)

    assert not (args.feature_mode and args.render_mode == "RGB+ED"), "Feature mode does not support depth rendering"
    # Initialize system state (RNG)
    args.eval = False
    safe_state(args.quiet)
    method_name = args.method_name
    print(args)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_mode, args.feature_npy, args.render_mode, method_name, args.normalize_flag) ###cnn
