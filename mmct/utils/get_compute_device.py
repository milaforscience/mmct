import torch


# %%
def get_compute_device(force_cpu=False):
    CUDA_available = torch.cuda.is_available()
    if force_cpu or not CUDA_available:
        return 'cpu'
    else:
        return 'cuda'
