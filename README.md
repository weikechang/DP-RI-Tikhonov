# Deep denoiser prior driven relaxed iterated Tikhonov method for low-count PET image restoration
# Abstract
Objective. Low-count positron emission tomography (PET) imaging is an efficient way to promote more widespread use of PET because of its short scan time and low injected activity. However, this often leads to low-quality PET images with clinical image reconstruction, due to high noise and blurring effects. Existing
PET image restoration (IR) methods hinder their own restoration performance due to the semi-convergence property and the lack of suitable denoiser prior. Approach. To overcome these limitations, we propose a novel deep plug-and-play IR method called
Deep denoiser Prior driven Relaxed Iterated Tikhonov method (DP-RI-Tikhonov). Specifically, we train a deep convolutional neural network (CNN) denoiser to generate a flexible deep denoiser prior to handle high noise. Then, we plug the deep
denoiser prior as a modular part into a novel iterative optimization algorithm to handle blurring effects and propose an adaptive parameter selection strategy for the iterative optimization algorithm. Main results. Simulation results show that the average normalized root mean square error (NRMSE) of DP-RI-Tikhonov has a 11.2% improvement at the optimal iteration and a 11.8% improvement at the early
stopping iteration compared to a conventional PET IR method, due to the deep denoiser prior. Thanks to the novel iterative optimization algorithm and adaptive parameter selection strategy, DP-RI-Tikhonov can obtain a near-optimal solution
through a simple early termination rule and has an average NRMSE reduction of at least 4.7% compared to other state-of-the-art deep plug-and-play IR methods. In addition, DP-RI-Tikhonov successfully reduces noise intensity and recovers fine details
in real experiments, leading to sharper and more uniform PET images compared to comparison methods. Significance. DP-RI-Tikhonov's ability to reduce noise intensity and effectively eliminate the semi-convergence property overcomes the limitations of existing methods, recovering high-quality images through a simple early termination rule. This advancement may have substantial implications for other medical IR.
## Demo ##
To run DP-RI-Tikhonov (RIT_DDP_rays.py), filling your path in "RIT_DDP_rays.py", and then run "python RIT_DDP_rays.py". 
## Restoration Visualization ##
![ezcv logo](https://github.com/weikechang/Deep-denoiser-prior-driven-relaxed-iterated-Tikhonov/blob/main/6.jpg)
## Quantitative Results ##
![ezcv logo](https://github.com/weikechang/Deep-denoiser-prior-driven-relaxed-iterated-Tikhonov/blob/main/results/results_SSIM_NRMSE.png)
## Contact ##
Should you have any question, please contact changweike@hust.edu.cn
## Solving your tasks ##
Deep denoiser prior driven relaxed iterated Tikhonov (DP-RI-Tikhonov) is a flexible image restoration method. You only need to replace our trained model with your trained model:
1. Replacing "../model_save/model_tlstage.pth" with your trained model.
2. Replacing Unet with your network, if your network is different from ours.


