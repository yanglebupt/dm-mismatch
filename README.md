# Article Title

This repo contains the official implementation for the paper XXXX.

by Le Yang (2019212184@bupt.edu.cn).

## Abstract

## Method
The proposed method uses $A_0$ (mmf 0 displacement known measurement matrix) and $A_{25}$ (mmf 25 displacement unknown measurement matrix) **Pre-Measure (PM)** image  separately to obtain $y_0$ and $y_1$. Use unknown measurement matrix $A_{25}$ measure unknown image $x$ to obtain $y_2$. Use the following algorithm to reconstruct the original image

$$
\begin{aligned}
   \arg\min\limits_{A_{recv}} ||y_2 - A_{recv}x||_2^2 \\ \\
   Pre-Measure(PM):  A_{recv1} = F(A_0, y_0, y_1; Speckle\_Measure(A, PM))\\ \\
   A_{recv} = A_{recv1} + F(A_0, y_0, y_2-Speckle\_Measure(A_{recv1}, x);  Speckle\_Measure(A, x)) \\ \\
   x'=G(y_2, A_{recv})
\end{aligned}
$$

![](https://latex.codecogs.com/svg.image?\arg\min\limits_{A_{recv}}||y_2-A_{recv}x||_2^2\\\\Pre-Measure(PM):A_{recv1}=F(A_0,y_0,y_1;Speckle\_Measure(A,PM))\\\\A_{recv}=A_{recv1}&plus;F(A_0,y_0,y_2-Speckle\_Measure(A_{recv1},x);Speckle\_Measure(A,x))\\\\x'=G(y_2,A_{recv}))

## Running Experiments

### Environment
We use the code experiment environment conditions as shown in the following list:

- PyTorch  1.11.0
- Python  3.8 (ubuntu20.04)
- Cuda  11.3
- RTX 2080 Ti (11GB) * 1

### Dependencies

You can install whatever python packages you find missing when running code, or just run the following command in the terminal.

```bash
pip install -r requirements.txt
```

### Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample] [--sample_cs_image]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --sample              Whether to produce samples from the model
  --sample_cs_image     Whether to produce cs image samples from the model
  --fast_fid            Whether to do fast fid test
  --resume_training     Whether to resume training
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  --ni                  No interaction. Suitable for Slurm Job launcher
  --dataset DATASET     dataset name for test

```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   └── lsun # some used LSUN files
│       │── bedroom_val_lmdb # just val datasets we don't used in our's experiment
│       │── church_outdoor_val_lmdb # just val datasets we don't used in our's experiment
│       └── tower_val_lmdb  # just tower_val_lmdb we used in our's experiment
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│       └── tower 
│          └── checkpoint_x.pth # download from google drive, the checkpoint file saved at the x-th training iteration
├── image_samples # contains generated samples
│   └── <i>
│       ├── <PMx>_<Name>_Mismatch_Recv.png # Reconstruction image uses PMx (Pre-Measure) image to solve Mismatch A_recv for compressive sensing with MMF measurement matrix
│       ├── <Name>_origin.png # Original image for compressive sensing with MMF measurement matrix
```

code about compressed sensing is in `cs_image/`. It includes the solution of mismatch problems for compressive sensing, multimode fiber speckle measurement matrix, and images from outside LSUN/tower dataset. They are structured as:

```bash
cs_image 
├── __init__.py       # Mismatch Problem Solve
├── measure_matrix.py # deterministic and random measurement matrix for compressive sensing
├── mmf_displacement  # download from google drive, measurement matrix composed of speckle patterns obtained from multimode fibers (MMF) with different displacements.  
│   ├── 0
│   │   ├── A_500_256_1.mat  # mmf speckle measurement matrix
│   │   ├── GI_x0y0.mat      # GI Original Image
│   │   ├── y_500_256_1.mat  # Experimental bucket detector value
│   │   └── y_original_500_256_1.mat # Original experimental bucket detector value (before sum)
│   ├── 10
│   │   ├── A_500_256_1.mat
│   │   ├── GI_x0y10.mat
│   │   ├── y_500_256_1.mat
│   │   └── y_original_500_256_1.mat
│   └── 25
│       ├── A_500_256_1.mat
│       ├── GI_x0y25.mat
│       ├── y_500_256_1.mat
│       └── y_original_500_256_1.mat
├── mmf_speckle.py # code to decoder mmf_displacement/<.mat> files
├── invert_sample.py  # sample for inverse problem used Bayes’ rule
├── svd_sample.py  # SNIPS's general_anneal_Langevin_dynamics which includes a singular value 
└── timg           # out-of LSUN/tower datasets images
    ├── Baboon.bmp
    ├── Barbara.bmp
    ├── Cameraman.bmp
    ├── Goldhill.bmp
    ├── Lena.bmp
    └── Peppers.bmp
```

And we modify `runners/ncsn_runner.py` to apply our method and store results into files, which in `../../autodl-tmp/` folder. You can access [dm-mismatch-results](https://github.com/yanglebupt/dm-mismatch-results) repo to obtain the more visualization results and code. We also implemented **OPM** and **GPSR** algorithms using Pytorch as a comparison with Diffusion Model in that repo. 

### Running

```bash
chmod -R u=rwx,g=rwx,o=rwx ./command.bash
./command.bash
```

Or

```bash
python main.py --config tower.yml --doc 0 --sample_cs_image --dataset tower
```

Samples will be saved in `<exp>/image_samples/images`.

You can uncomment some selective code in `runners/ncsn_runner.py` to view different results. For example, the following uses different images for `--sample_cs_image`.

```python
# line 274
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
```

Or use different measure_matrixs for `--sample`.

```python
# line 136
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
    get_measure_matrix = get_gussian_measure_matrix  # to change different measure_matrix
```

## Data availability
Different displacement MMF measurement matrixs at https://drive.google.com/drive/folders/1_RlwkPU6pSR6FRqWL7TT7ovwtphenpcy?usp=drive_link. Download and place into `cs_image/mmf_displacement` folder.

## Pretrained Checkpoints

Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing

These checkpoint files are provided as-is from the authors of [NCSNv2](https://github.com/ermongroup/ncsnv2). We just use the **LSUN-tower datasets' pretrained checkpoints**. We assume the `--exp` argument is set to `exp`. Download and place into `<exp:exp>/logs/<doc:0>/tower` folder.

## Acknowledgement

This repo is largely based on the [NCSNv2](https://github.com/ermongroup/ncsnv2) repo, and uses code from [SNIPS](https://github.com/bahjat-kawar/snips_torch/blob/main/models/__init__.py) for implementing the SNIPS's general_anneal_Langevin_dynamics sample which includes a singular value decomposition (SVD) of the degradation operator.

## References

If you find the code/idea useful for your research, please consider citing

```bib
```

