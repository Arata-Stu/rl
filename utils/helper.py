from typing import Union
import numpy as np
import torch 
import torch.nn as nn


def numpy_to_tensor(input: np.ndarray, device: torch.device) -> torch.Tensor:
    ## input: np.ndarray (H W C), device: torch.device
    ## output: torch.Tensor (C H W)
    output = torch.tensor(input, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return output

def tensor_to_numpy(input: torch.Tensor) -> np.ndarray:
    ## input: torch.Tensor (B C H W)
    ## output: np.ndarray (B H W C)
    output = input.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return output
    
def encode_img(encoder: nn.Module, img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(img, np.ndarray):
        img = numpy_to_tensor(img, encoder.device)

    z = encoder.encode(img)

    z = z.detach().cpu().numpy()
    return z

def decode_img(encoder: nn.Module, z: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(z, np.ndarray):
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(encoder.device)
    img = encoder.decode(z)
    img = tensor_to_numpy(img)
    return img

def transform_action(raw_action, action_space):
    """
    汎用的なアクション変換関数。
    
    Args:
        raw_action (np.ndarray or torch.Tensor): Actor から出力された [-1,1] の範囲のアクション
        action_space (gymnasium.spaces.Box): 環境の action_space
    
    Returns:
        np.ndarray: 環境に適用できるアクション
    """
    low = action_space.low
    high = action_space.high
    
    # [-1,1] -> [low, high] にスケーリング
    transformed_action = low + (raw_action + 1.0) * 0.5 * (high - low)
    
    return transformed_action


# soft_update 関数
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
