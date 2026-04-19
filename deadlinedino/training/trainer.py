import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import os
import time
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import json

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify
from .. import utils
from . import schedule_utils # <-- IMPORT THE NEW SCHEDULER

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    if lp.source_type=="colmap":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution
    elif lp.source_type=="slam":
        cameras_info,camera_frames,init_xyz,init_color=io_manager.load_slam_result(lp.source_path)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    # Load train/test split from train_test_split.json
    split_json_path = os.path.join(lp.source_path, "train_test_split.json")
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        train_image_names = set(split_data.get("train", []))
        test_image_names = set(split_data.get("test", []))

        # Filter frames based on the split
        training_frames = [c for c in camera_frames if c.name in train_image_names]
        test_frames = [c for c in camera_frames if c.name in test_image_names] if lp.eval else None
        print(f"Loaded train/test split: {len(training_frames)} training images, {len(test_frames) if test_frames else 0} test images")
    else:
        # Fallback to original logic if train_test_split.json doesn't exist
        if lp.eval:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
        else:
            training_frames=camera_frames
            test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    # --- START: New Scheduler/Optimizer Logic ---
    
    # 1. Get original images for FFT analysis (must be on CUDA)
    # We load them onto CUDA here and then free them after scheduler init
    original_images_for_fft = []
    if pp.resolution_mode == "freq":
        for frame in training_frames:
            # --- CORRECT FIX: Access the dictionary with the lp.resolution key ---
            # This tensor is already on the GPU from the CameraFrameDataset init
            original_images_for_fft.append(frame.image[lp.resolution])

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    
    # 2. Initialize the Training Scheduler
    # --- MODIFIED: Pass 'dp' as the second argument ---
    scheduler = schedule_utils.TrainingScheduler(op, dp, pp, init_points_num, original_images_for_fft)

    # 3. Free the image tensors, we don't need them anymore
    del original_images_for_fft
    torch.cuda.empty_cache()

    # 4. Get the dynamic decay_from_iter value from the scheduler
    decay_from_iter = scheduler.lr_decay_from_iter()

    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        
        # 5. Pass the dynamic decay_from_iter to the optimizer
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp,
                                              decay_from_iter=decay_from_iter) # <-- PASS VALUE
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        # TODO: Need to handle re-setting the scheduler's decay_from_iter on checkpoint load
        # For now, we assume it's correct from the initial load
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    
    # --- END: New Scheduler/Optimizer Logic ---
    
    actived_sh_degree=0

    #learnable view matrix
    if op.learnable_viewproj:
        view_params=[np.concatenate([frame.qvec,frame.tvec])[None,:] for frame in trainingset.frames]
        view_params=torch.tensor(np.concatenate(view_params),dtype=torch.float32,device='cuda')
        view_params=torch.nn.Embedding(view_params.shape[0],view_params.shape[1],_weight=view_params,sparse=True)
        camera_focal_params=torch.nn.Parameter(torch.tensor(trainingset.cameras[0].focal_x,dtype=torch.float32,device='cuda'))#todo fix multi cameras
        view_opt=torch.optim.SparseAdam(view_params.parameters(),lr=1e-4)
        proj_opt=torch.optim.Adam([camera_focal_params,],lr=1e-5)

    #init
    total_epoch=int(op.iterations/len(trainingset))
    
    # --- START: Global Step Counter ---
    total_iterations = op.iterations 
    global_step = start_epoch * len(train_loader) 
    # --- END: Global Step Counter ---

    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    
    # Note: density_controller is now initialized *after* the scheduler
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    # Time-based stopping: track start time for 59.5 second timeout
    training_start_time = time.time()
    elapsed_time = 0.0
    for epoch in range(start_epoch,total_epoch):

        old_elapsed_time = elapsed_time
        elapsed_time = time.time() - training_start_time
        epoch_time = elapsed_time - old_elapsed_time
        if elapsed_time >= 60 - epoch_time - 0.5:  # Leave a small buffer
            progress_bar.close()

            # Save the most recent .ply file
            save_path = os.path.join(lp.model_path, "point_cloud", f"timeout_epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)

            # Save training time as JSON
            metrics = {
                "time": elapsed_time,
                "model_path": lp.model_path,
                "status": "timeout",
                "final_epoch": epoch,
                "total_epochs": total_epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metrics_file_path = os.path.join(save_path, "training_metrics.json")
            with open(metrics_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save .ply file
            if pp.cluster_size:
                tensors = scene.cluster.uncluster(xyz, scale, rot, sh_0, sh_rest, opacity)
            else:
                tensors = xyz, scale, rot, sh_0, sh_rest, opacity
            param_nyp = []
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path, "point_cloud.ply"), *param_nyp)
            if op.learnable_viewproj:
                torch.save(list(view_params.parameters())+[camera_focal_params], os.path.join(save_path, "viewproj.pth"))

            print(f"Saved checkpoint to {save_path}")
            print(f"Training metrics saved to {metrics_file_path}")
            break

        with torch.no_grad():
            if pp.cluster_size>0 and (epoch-1)%dp.densification_interval==0:
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image,idx in train_loader:
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                
                # --- START: New Coarse-to-Fine Logic ---
                
                # 1. Get render_scale from the scheduler
                render_scale = scheduler.get_res_scale(global_step)

                #DEBUG
                # print(f"Global Step: {global_step}, Render Scale: {render_scale}")
                
                # 2. Load GT image and downsample it
                gt_image=gt_image.cuda()/255.0 
                
                if render_scale > 1:
                    gt_image = torch.nn.functional.interpolate(
                        gt_image, # Pass 4D tensor directly
                        scale_factor = 1.0 / render_scale, 
                        mode="bilinear", 
                        recompute_scale_factor=True, 
                        antialias=True
                    )
                
                # --- END: New Coarse-to-Fine Logic ---
                
                if op.learnable_viewproj:
                    #fix view matrix
                    view_param_vec=view_params(idx.cuda())
                    qvec=torch.nn.functional.normalize(view_param_vec[:,:4],dim=1)
                    tvec=view_param_vec[:,4:]
                    rot_matrix=utils.wrapper.CreateTransformMatrix.call_fused(torch.ones((3,qvec.shape[0]),device='cuda'),qvec.transpose(0,1).contiguous())
                    view_matrix[:,:3, :3] = rot_matrix.permute(2,0,1)
                    view_matrix[:,3, :3] = tvec

                    #fix proj matrix
                    focal_x=camera_focal_params
                    focal_y=camera_focal_params*gt_image.shape[3]/gt_image.shape[2]
                    proj_matrix[:,0,0]=focal_x
                    proj_matrix[:,1,1]=focal_y

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)
                
                #DEBUG
                # print(f"img size: {img.shape}, gt_image size: {gt_image.shape}, render_scale: {render_scale}")
                
                # Add batch dimension to img to match gt_image
                img_b = img.unsqueeze(0) 
                
                l1_loss=__l1_loss(img_b, gt_image) 
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img_b, gt_image)
                
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                    # proj_opt.step()
                    # proj_opt.zero_grad()
                schedular.step()

                global_step += 1

        if epoch in test_epochs:
            with torch.no_grad():
                _cluster_origin=None
                _cluster_extend=None
                if pp.cluster_size:
                    _cluster_origin,_cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for view_matrix,proj_matrix,frustumplane,gt_image,idx in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0

                        if name=="Trainingset" and op.learnable_viewproj:
                            view_param_vec=view_params(idx.cuda())
                            qvec=torch.nn.functional.normalize(view_param_vec[:,:4],dim=1)
                            tvec=view_param_vec[:,4:]
                            rot_matrix=utils.wrapper.CreateTransformMatrix.call_fused(torch.ones((3,qvec.shape[0]),device='cuda'),qvec.transpose(0,1).contiguous())
                            view_matrix[:,:3, :3] = rot_matrix.permute(2,0,1)
                            view_matrix[:,3, :3] = tvec

                            focal_x=camera_focal_params
                            focal_y=camera_focal_params*gt_image.shape[3]/gt_image.shape[2]
                            proj_matrix[:,0,0]=focal_x
                            proj_matrix[:,1,1]=focal_y

                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(_cluster_origin,_cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,normal,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {}".format(epoch,name,torch.concat(psnr_list,dim=0).mean()))

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)
        progress_bar.update()

        # if epoch in save_ply or epoch==total_epoch-1:
        #     if epoch==total_epoch-1:
        #         progress_bar.close()
        #         print("{} takes: {} s".format(lp.model_path,progress_bar.format_dict['elapsed']))
        #         save_path=os.path.join(lp.model_path,"point_cloud","finish")
        #     else:
        #         save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))    

        #     if pp.cluster_size:
        #         tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
        #     else:
        #         tensors=xyz,scale,rot,sh_0,sh_rest,opacity
        #     param_nyp=[]
        #     for tensor in tensors:
        #         param_nyp.append(tensor.detach().cpu().numpy())
        #     io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)
        #     if op.learnable_viewproj:
        #         torch.save(list(view_params.parameters())+[camera_focal_params],os.path.join(save_path,"viewproj.pth"))
        
        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                elapsed_time = progress_bar.format_dict['elapsed']
                print("{} takes: {} s".format(lp.model_path, elapsed_time))
                save_path=os.path.join(lp.model_path,"point_cloud","finish")
            else:
                elapsed_time = time.time() - training_start_time
                save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))

            # Create directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save training time as JSON
            metrics = {
                "time": elapsed_time,
                "model_path": lp.model_path,
                "status": "completed" if epoch == total_epoch-1 else "checkpoint",
                "epoch": epoch,
                "total_epochs": total_epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(save_path, "training_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save .ply file
            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)
            if op.learnable_viewproj:
                torch.save(list(view_params.parameters())+[camera_focal_params],os.path.join(save_path,"viewproj.pth"))
            
            print(f"Training metrics saved to {os.path.join(save_path, 'training_metrics.json')}")

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    return