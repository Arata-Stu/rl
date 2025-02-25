import torch
import torchvision.utils as vutils
import numpy as np
import os

def visualize_and_save_reconstruction(original: np.ndarray, reconstructed: np.ndarray, output_dir, step):
    """
    オリジナル画像と再構築画像を1つの画像として保存する関数 (torchvisionを使用)

    Args:
        original (np.ndarray): 元画像 (H, W, C) の NumPy 配列 (0~1 or 0~255)
        reconstructed (np.ndarray): 再構築画像 (H, W, C) の NumPy 配列 (0~1 or 0~255)
        output_dir (str): 画像を保存するディレクトリ
        step (int): 現在のステップ数（連番保存用）
    """

    # NumPy (H, W, C) → Torch (C, H, W) に変換
    original_tensor = torch.from_numpy(original).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    reconstructed_tensor = torch.from_numpy(reconstructed).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    ## 0~1にクリップ
    original_tensor = torch.clamp(original_tensor, 0, 1)
    reconstructed_tensor = torch.clamp(reconstructed_tensor, 0, 1)

    # 画像を並べるためのリスト
    images = torch.stack([original_tensor, reconstructed_tensor], dim=0)  # (N, C, H, W)

    # 保存ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像を保存
    output_path = os.path.join(output_dir, f"reconstruction_step_{step:04d}.png")
    vutils.save_image(images, output_path, nrow=2, normalize=True)

