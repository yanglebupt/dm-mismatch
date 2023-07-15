import numpy as np
import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation
import h5py

import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation)
from models import get_sigmas
from models.ema import EMAHelper
from PIL import Image
from cs_image import *
import time
from scipy import io

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
                    
    def get_pretrained_score(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, self.args.dataset, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, self.args.dataset, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                              map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()
        
        states[0]["module.sigmas"] = sigmas_th
        score.load_state_dict(states[0], strict=False)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            
        score.eval()
        
        return score,sigmas
    
    def save_imgs(self,all_samples,refer_images,restore_last=True,save_mid_img=False,sample_number=1):
        imgs = []
        if restore_last:
            sample=all_samples[-1]
            sample = sample.view(sample.shape[0], self.config.data.channels,
                                    self.config.data.image_size,
                                    self.config.data.image_size)
            sample = inverse_data_transform(self.config, sample)

            image_grid = make_grid(sample, int(np.sqrt(sample_number)))
            save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_cs_image_fin.png'))    
        else:
            for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                      desc="saving image samples"):
                sample = sample.view(sample.shape[0], self.config.data.channels,
                                    self.config.data.image_size,
                                    self.config.data.image_size)

                sample = inverse_data_transform(self.config, sample)

                image_grid = make_grid(sample, int(np.sqrt(sample_number)))

                if save_mid_img:
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_cs_image_{}.png'.format(i)))

                if i==len(all_samples)-1:
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_cs_image_fin.png'))

                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)
            imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
        
        image_grid = make_grid(refer_images, int(np.sqrt(sample_number)))
        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_cs_image_origin.png'))
        
    def recv_unknow_A(self, y_1, y, measure_matrix, speckle_measure, idxs = 10, error_y=None, A_2=None):
        know_y = y_1[:,0,...].unsqueeze(1)
        unknow_y = y[:,0,...].unsqueeze(1)
        error_y = y[:,0,...].unsqueeze(1) if error_y is None else error_y
        A_2 = torch.zeros(measure_matrix.shape).to(self.config.device) if A_2 is None else A_2
        idx = 0
        epoch_errors=[]
        while idx <= idxs:
            # 计算未知观测下的观测矩阵
            A_2 += predict_unknown_measure_matrix_float32(measure_matrix, know_y, error_y)
            recv_y = speckle_measure(A_2)
            print("========error==========")
            print(error_y)
            print(recv_y)
            error_y=unknow_y-recv_y
            e = abs(error_y.cpu().numpy()).mean()
            print(e)
            epoch_errors.append(e)
            idx+=1

        print("========predict==========")
        print(A_2)
        print(y)
        recv_y = speckle_measure(A_2)
        print(recv_y)
        print("========error==========")
        print(y-recv_y)
        return A_2, epoch_errors
    
    def sample(self):
        """
        get_gussian_measure_matrix
        get_bernoulli_measure_matrix
        get_random_svd_compose_measure_matrix
        get_sparse_random_measure_matrix
        get_toeplitz_loop_measure_matrix
        
        get_hada_measure_matrix
        get_fft_measure_matrix
        """
        measure_rate = self.config.cs_image.measure_rate
        print(measure_rate)
        get_measure_matrix = get_gussian_measure_matrix
        f=h5py.File("../../autodl-tmp/common_res/no_train_0.15.h5", "a")
        
        score,sigmas = self.get_pretrained_score()
        _, dataset = get_dataset(self.args, self.config, noTrain=True)
        recv_imgs = self.sample_image(dataset, measure_rate, get_measure_matrix,score,sigmas,save_img=True)
        f.create_dataset("sde", data=recv_imgs)
        f.close()
        
    def sample_image(self, dataset, measure_rate, get_measure_matrix, score,sigmas, 
                                    restore_last=True, save_img=False, save_mid_img=False):
        
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=False,
                        num_workers=4)
        nums = len(dataset)
        recv_imgs=None
       
        N = self.config.data.image_size ** 2
        M = int(N * measure_rate)
        measure_sigma = self.config.cs_image.measure_sigma
        measure_matrix = get_measure_matrix(self.config.data.channels,M,N,
                                              device=self.config.device,sigma=measure_sigma,channel_common=True)
        print(measure_matrix, measure_matrix.shape)
        
        nosie_sigma=self.config.cs_image.nosie_sigma
        nosie = self.config.cs_image.nosie
        sample_number = 6
        sigma_0=self.config.cs_image.sigma_0
        
        names=[
             "Baboon",
             "Peppers",
             "Goldhill",
             "Barbara",
             "Cameraman",
             "Lena"
        ]
        refer_images = torch.empty(sample_number, 1,
                                        self.config.data.image_size,
                                        self.config.data.image_size)
        for i in range(6):
            img = get_t_ori_image(names[i], self.config.device, self.config.data.image_size, self.config.data.image_size)
            refer_images[i,...] = img[0]
        
        # for i, (refer_images,_) in enumerate(dataloader):
        #     break
            
        refer_images = refer_images.to(self.config.device)        
        y = speckle_measure(refer_images, measure_matrix, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            
        samples = torch.randn(sample_number, self.config.data.channels,
                                        self.config.data.image_size,
                                        self.config.data.image_size, device=self.config.device)
        samples = data_transform(self.config, samples)
        all_samples = general_anneal_Langevin_dynamics(measure_matrix, y, sigma_0, samples, score, sigmas,
                                                      self.config.sampling.n_steps_each,
                                                      self.config.sampling.step_lr, verbose=True,
                                                      nosie=nosie,
                                                      denoise=self.config.sampling.denoise,
                                                      final_only=restore_last,
                                                      ode = self.config.cs_image.ode)
        sample = inverse_data_transform(self.config, all_samples[-1])
        batch_res=sample.numpy()
        recv_imgs = batch_res if recv_imgs is None else np.concatenate([recv_imgs,batch_res],axis=0) 

        if not save_img:
            return recv_imgs

        self.save_imgs(all_samples, refer_images, restore_last=restore_last, save_mid_img=save_mid_img)
        return recv_imgs
        
        
    def sample_cs_image(self):

        measure_rate = self.config.cs_image.measure_rate
        get_measure_matrix = get_mmf_speckle_measure_matrix
        # f=h5py.File("../../autodl-tmp/diff_pre_images/recv_unknown_no_train_tt.h5", "a")
        
        score,sigmas = self.get_pretrained_score()
        _, dataset = get_dataset(self.args, self.config, noTrain=True)
        """
        sample_cs_image_one_dataset_dismatch_update
        sample_cs_image_one_dataset_dismatch
        """
        recv_imgs = self.sample_cs_image_one_dataset_dismatch_update(dataset, measure_rate, get_measure_matrix,score,sigmas,save_img=True, h5F=None)
        # f.create_dataset("Img_0", data=recv_imgs)
        # f.close()
        
        
    def sample_cs_image_one_dataset_dismatch_update(self, dataset, measure_rate, get_measure_matrix, score,sigmas, 
                                    restore_last=True, save_img=False, save_mid_img=False, h5F=None):
        
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=False,
                        num_workers=4)
        nums = len(dataset)
        recv_imgs=None
       
        N = self.config.data.image_size ** 2
        M = int(N * measure_rate)
        
        nosie_sigma=self.config.cs_image.nosie_sigma
        nosie = self.config.cs_image.nosie
        measure_sigma = self.config.cs_image.measure_sigma
        sample_number = 1
        sigma_0=self.config.cs_image.sigma_0
        device=self.config.device
        
        # 恢复时已知的观测矩阵
        measure_matrix, _ = get_mmf_speckle_measure_matrix(0, self.config.data.channels,M,N,
                                              device=self.config.device,sigma=measure_sigma,channel_common=True)
        # 未知的观测矩阵
        measure_matrix_unkonw, _ = get_mmf_speckle_measure_matrix(25, self.config.data.channels,M,N,
                                              device=self.config.device,sigma=measure_sigma,channel_common=True)
        
        
        for i, (refer_images,_) in enumerate(dataloader):
            """
            get_t_ori_image
             - Baboon
             - Peppers
             - Goldhill
             - Barbara
             - Cameraman
             - Lena
            """
            # 预图像
            # refer_images_2 = get_mmf_ori_image(0, self.config.device, self.config.data.image_size, self.config.data.image_size)  
            # refer_images_2 = refer_images[0,0,...].unsqueeze(0).unsqueeze(1).to(self.config.device)   
            refer_images_2 = get_pre_measure_img(self.config.data.image_size,
                                self.config.data.image_size).unsqueeze(0).unsqueeze(1).to(self.config.device)
            
            # 未知图像
            name="Baboon"
            # refer_images_3 = get_t_ori_image(name, self.config.device, self.config.data.image_size, self.config.data.image_size) 
            refer_images_3 = get_mmf_ori_image(0, self.config.device, self.config.data.image_size, self.config.data.image_size)  

            y_2 = speckle_measure(refer_images_2, measure_matrix, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            y_3 = speckle_measure(refer_images_3, measure_matrix, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            
            u_y_2 = speckle_measure(refer_images_2, measure_matrix_unkonw, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            u_y_3 = speckle_measure(refer_images_3, measure_matrix_unkonw, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            
            
            start_time = time.time()
            n_y = y_2
            match_images=refer_images_2
            A_2, _ = self.recv_unknow_A(
                y_2,
                u_y_2,
                measure_matrix,
                lambda A_2 : speckle_measure(match_images, A_2, nosie_sigma=nosie_sigma, nosie=nosie, channel_common=True),
                idxs=10
            )
            
            
            refer_images = refer_images_3
            y = u_y_3
            
            recv_y = speckle_measure(refer_images, A_2, nosie_sigma=nosie_sigma, nosie=nosie, channel_common=True)
            error_y = y - recv_y
            
            A_2, epoch_errors = self.recv_unknow_A(
                n_y,
                y,
                measure_matrix,
                lambda A_2 : speckle_measure(refer_images, A_2, nosie_sigma=nosie_sigma, nosie=nosie, channel_common=True),
                error_y=error_y,
                A_2=A_2,
                idxs=20
            )
            end_time = time.time()
            
            # x = refer_images_3.cpu().numpy()
            # y = y.cpu().numpy()
            # A_2 = A_2.cpu().numpy()
            # io.savemat('../../autodl-tmp/recvs/PM2/{}.mat'.format(name), {'x': x, 
            #                                                           'y': y, 
            #                                                           "A_recv": A_2})
            # if h5F:
                # h5F.create_dataset("GI_50_Error", data=np.array([epoch_errors][-1]))
                # h5F.create_dataset("Error", data=np.array(epoch_errors))
                # h5F.create_dataset("Time", data=np.array([end_time - start_time]))
            
            samples = torch.randn(sample_number, self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_size, device=self.config.device)
            samples = data_transform(self.config, samples)

            all_samples = general_anneal_Langevin_dynamics(A_2, y, sigma_0, samples, score, sigmas,
                                                          self.config.sampling.n_steps_each,
                                                          self.config.sampling.step_lr, verbose=True,
                                                          nosie=nosie,
                                                          denoise=self.config.sampling.denoise,
                                                          final_only=restore_last,
                                                          ode = self.config.cs_image.ode)
            sample = inverse_data_transform(self.config, all_samples[-1])
            batch_res=sample.numpy()
            recv_imgs = batch_res if recv_imgs is None else np.concatenate([recv_imgs,batch_res],axis=0) 
            
            if not save_img:
                return recv_imgs
            
            break

        self.save_imgs(all_samples, refer_images, restore_last=restore_last, save_mid_img=save_mid_img)
        return recv_imgs

    def sample_cs_image_one_dataset_dismatch(self, dataset, measure_rate, get_measure_matrix, score,sigmas, 
                                    restore_last=True, save_img=False, save_mid_img=False,h5F=None):
        
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=False,
                        num_workers=4)
        nums = len(dataset)
        recv_imgs=None
       
        N = self.config.data.image_size ** 2
        M = int(N * measure_rate)
        measure_sigma = self.config.cs_image.measure_sigma
        # 恢复时已知的观测矩阵
        measure_matrix, _ = get_measure_matrix(0, self.config.data.channels,M,N,
                                              device=self.config.device,sigma=measure_sigma,channel_common=True)
        
        # 未知的观测矩阵
        measure_matrix_unkonw, _ = get_measure_matrix(25, self.config.data.channels,M,N,
                                              device=self.config.device,sigma=measure_sigma,channel_common=True)
        print(measure_matrix_unkonw, measure_matrix_unkonw.shape)
        
        nosie_sigma=self.config.cs_image.nosie_sigma
        nosie = self.config.cs_image.nosie
        sample_number = 1
        sigma_0=self.config.cs_image.sigma_0
        
        
        for i, (refer_images,_) in enumerate(dataloader):
            """
            get_mmf_ori_image
            get_t_ori_image
             - Baboon
             - Peppers
             - Goldhill
             - Barbara
             - Cameraman
             - Lena
            """
            # refer_images = get_mmf_ori_image(0, self.config.device, self.config.data.image_size, self.config.data.image_size)
            refer_images = refer_images[0,0,...].unsqueeze(0).unsqueeze(1).to(self.config.device)    
            # refer_images = get_t_ori_image("Lena", self.config.device, self.config.data.image_size, self.config.data.image_size)
            
            """
            speckle_measure
            get_mmf_measure
            """
            # 已知观测矩阵生成的 y_1
            y_1 = speckle_measure(refer_images, measure_matrix, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            # 未知观测矩阵生成的 y
            y = speckle_measure(refer_images, measure_matrix_unkonw, nosie_sigma=nosie_sigma,nosie=nosie, channel_common=True)
            
            
            A_2,_ = self.recv_unknow_A(
                y_1,
                y,
                measure_matrix,
                lambda A_2 : speckle_measure(refer_images, A_2, nosie_sigma=nosie_sigma, nosie=nosie, channel_common=True)
            )
            
            #### just for GI ###
            # f=h5py.File("../../autodl-tmp/recvs/A_1_A_2_GI_A_recv.h5", "w")
            # f.create_dataset("A_1", data = measure_matrix.cpu().numpy())
            # f.create_dataset("A_2", data = measure_matrix_unkonw.cpu().numpy())
            # f.create_dataset("GI_Recv_A_2", data = A_2.cpu().numpy())
            # f.create_dataset("GI_y_1", data = y_1.cpu().numpy())
            # f.create_dataset("GI_y_2", data = y.cpu().numpy())
            # f.close()
            # return
            
            samples = torch.randn(sample_number, self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_size, device=self.config.device)
            samples = data_transform(self.config, samples)

            """
            general_anneal_Langevin_dynamics
            inverse_anneal_Langevin_dynamics
            """
            """
            measure_matrix --> y_1              konwn_0
            measure_matrix_unkonw --> y         konwn_25
            measure_matrix  --> y               dismatch
            A_2 --> y                           dismatch proposed
            """
            all_samples = general_anneal_Langevin_dynamics(A_2, y, sigma_0, samples, score, sigmas,
                                                          self.config.sampling.n_steps_each,
                                                          self.config.sampling.step_lr, verbose=True,
                                                          nosie=nosie,
                                                          denoise=self.config.sampling.denoise,
                                                          final_only=restore_last,
                                                          ode = self.config.cs_image.ode)
            sample = inverse_data_transform(self.config, all_samples[-1])
            batch_res=sample.numpy()
            recv_imgs = batch_res if recv_imgs is None else np.concatenate([recv_imgs,batch_res],axis=0) 
            
            if not save_img:
                return recv_imgs
            
            break

        self.save_imgs(all_samples, refer_images, restore_last=restore_last, save_mid_img=save_mid_img)
        return recv_imgs