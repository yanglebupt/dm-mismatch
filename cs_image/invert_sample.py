import torch
import numpy as np
from cs_image.measure_matrix import speckle_measure,unsqueeze

"""
逆问题的 anneal_Langevin_dynamics
A: [C] M N
y: B C M
x_mode: B C H W
nosie_sigma: float
"""
@torch.no_grad()
def inverse_anneal_Langevin_dynamics(A, y, nosie_sigma, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=True, nosie=False, denoise=True,ode=False):
    images = []
    errors=[]
    isBreak=False
        
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            if isBreak:
                break
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                # ============================
                B,C,H,W = x_mod.shape
                meas = speckle_measure(x_mod, A, nosie_sigma, nosie=False, channel_common=False)
                meas_grad = torch.matmul( torch.transpose(unsqueeze(A), 2, 3), (y - meas).unsqueeze(3))    # [1 C M N] * [B C M 1] = [B C N 1]
                nosie_sigma = nosie_sigma if nosie else 0
                meas_grad = meas_grad.reshape(B,C,H,W)
                
                grad_norm = torch.norm(grad.view(grad.shape[0], C, -1), dim=-1).mean()   # up
                meas_grad_norm = torch.norm(meas_grad.view(meas_grad.shape[0], C, -1), dim=-1).mean()  # down
                gamma = meas_grad_norm / grad_norm
                
                meas_grad /= (torch.norm( meas_grad ) / torch.norm( grad ) + nosie_sigma ** 2)
                # meas_grad /= (gamma + nosie_sigma ** 2)
                # ============================

                noise = torch.randn_like(x_mod)
                
                # if s==n_steps_each-1 and c==len(sigmas)-1:
                #     x_mod = x_mod + step_size * (grad + meas_grad)
                # else:
                #     x_mod = x_mod + step_size * (grad + meas_grad) + noise * np.sqrt(step_size * 2)
                
                if ode:
                    x_mod = x_mod + step_size * (grad + meas_grad)
                else:
                    x_mod = x_mod + step_size * (grad + meas_grad) + noise * np.sqrt(step_size * 2)
                
                error = torch.norm((meas-y).squeeze(), dim=-1).mean()
                errors.append(error.item())

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, sigma:{} ,step_size: {}, error: {}, grad_norm:{}, meas_grad_norm:{}, gamma:{}".format(
                        c, sigma, step_size, error.item(), grad_norm.item(), meas_grad_norm.item(), gamma.item() ))
                    
                if np.isnan((meas - y).norm().cpu().numpy()):
                    isBreak = True
                    break

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)    
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images
