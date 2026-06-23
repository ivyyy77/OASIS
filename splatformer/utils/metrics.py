import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch
import lpips
import json

class MetricComputer:
    def __init__(self, forward_bs=8):
        lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to('cuda')
        self.metrics = {
            'psnr': lambda x,y: psnr(x,y).squeeze(), #(N,)
            'ssim': lambda x, y: ssim(x.permute(0,3,1,2),y.permute(0,3,1,2), window_size=11, size_average=False), #(N,)
            'lpips': lambda x,y: lpips_fn(x.permute((0,3,1,2)),y.permute(0,3,1,2),normalize=True).squeeze() #(N,)
        }
        self.results = {metric: [] for metric in self.metrics.keys()}
        self.results_dict = {}
        self.forward_bs = 16

    def update(self, img1s, img2s, name):
        if name not in self.results_dict:
            self.results_dict[name] = {}
        if img1s.max() > 1: #255
            img1s = img1s/255.0
        if img2s.max() > 1:
            img2s = img2s/255.0
        for metric, fn in self.metrics.items():
            if metric=='lpips':
                # Concerning OOD issue, we need to split imgs into batches
                total_n = img1s.size(0)
                n_batches = math.ceil(total_n/self.forward_bs)
                for i in range(n_batches): #But later I found that this is not the issue
                    start = i*self.forward_bs
                    end = min((i+1)*self.forward_bs, total_n)
                    res = fn(img1s[start:end], img2s[start:end])
                    if res.dim()==0:
                        res = res.unsqueeze(0)
                    self.results[metric].append(res)
            else:
                res = fn(img1s, img2s)
                if res.dim()==0:
                    res = res.unsqueeze(0)
                self.results[metric].append(res)
            # turn 0-dim to 1-dim
            self.results_dict[name][metric] = [r.item() for r in res] #self.results[metric][-1].mean().item() #?? (per-scene) (per-img?)


    def update_value(self, key, value, name):
        if key in self.results:
            self.results[key].append(value)
        else:
            self.results[key] = [value]
        if name not in self.results_dict:
            self.results_dict[name] = {}
        self.results_dict[name][key] = value.item()

    def sum(self):
        self.summation = {}
        for metric in self.results.keys():
            if max([x.dim() for x in self.results[metric]]) == 0:
                self.summation[metric] = sum(self.results[metric])
            else:
                self.summation[metric] = torch.cat(self.results[metric]).sum()
        return self.summation
    
    def concat(self):
        self.concatenation = {}
        for metric in self.results.keys():
            self.concatenation[metric] = torch.cat(self.results[metric])
        return self.concatenation
    
    def finalize(self):
        #compute the mean of the results
        self.reduced = {}
        for metric in self.results.keys():
            self.reduced[metric] = torch.cat(self.results[metric]).mean().item()
        return self.reduced
    
    def write_to_file(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.results_dict, f, indent=4)
    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, m=None):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True, keep_featuremap=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if keep_featuremap:
        return ssim_map.mean(1) # [bs, H, W]
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    