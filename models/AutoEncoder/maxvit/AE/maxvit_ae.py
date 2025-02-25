import sys
sys.path.append('../..')
import torch
import torch.nn as nn
from omegaconf import DictConfig
from ..timm.maxvit_encoder import MaxxVitEncoder
from ..timm.maxvit_decoder import MaxxVitDecoder

class MaxVITAE(nn.Module):
    """
    MaxVIT Autoencoder (AE) のクラス実装。

    画像を潜在ベクトルに圧縮し、リプレイバッファに保存しやすくする。
    潜在ベクトルは Flatten (1D) して保存し、デコード時に Reshape する。
    """
    def __init__(self, encoder_cfg: DictConfig, latent_dim: int = 512, device: torch.device = 'cpu'):
        super(MaxVITAE, self).__init__()
        
        self.device = device
        # YAML設定を読み込み
        encoder_cfg = encoder_cfg
        decoder_cfg = encoder_cfg

        # エンコーダの初期化
        self.encoder = MaxxVitEncoder(encoder_cfg.maxvit, img_size=encoder_cfg.maxvit.img_size)

        # エンコーダの出力チャネル数・特徴マップサイズを取得
        encoder_out_chs = self.encoder.feature_info[-1]['num_chs']
        encoder_out_size = encoder_cfg.maxvit.img_size // self.encoder.feature_info[-1]['reduction']
        self.latent_shape = (encoder_out_chs, encoder_out_size, encoder_out_size)  # 特徴マップの形状を保存

        # Flatten (特徴マップ → 1D 潜在ベクトル)
        self.flatten = nn.Flatten()

        # 潜在ベクトルの次元数を取得
        latent_dim_from_encoder = encoder_out_chs * encoder_out_size * encoder_out_size
        self.latent_dim = min(latent_dim, latent_dim_from_encoder)  # 指定の次元数 or エンコーダ出力のサイズ

        # 圧縮層: `C * H * W` → `latent_dim`
        self.fc_encode = nn.Linear(latent_dim_from_encoder, self.latent_dim)
        
        # デコーダの入力層: `latent_dim` → `C * H * W`
        self.fc_decode = nn.Linear(self.latent_dim, latent_dim_from_encoder)

        # デコーダの初期化
        self.decoder = MaxxVitDecoder(decoder_cfg.maxvit, in_chans=encoder_out_chs, input_size=encoder_out_size)

        ## ckptの読み込み
        
        if encoder_cfg.ckpt_path is not None:
            print(f"Load checkpoint from {encoder_cfg.ckpt_path}")
            ckpt = torch.load(encoder_cfg.ckpt_path, map_location=device)
            
            state_dict = ckpt['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                # "model." がキーの先頭にある場合は取り除く
                new_key = key.replace("model.", "")
                new_state_dict[new_key] = value
            self.load_state_dict(new_state_dict)

    def encode(self, x):
        """
        入力 x をエンコードし、Flatten して潜在ベクトルに変換
        """
        z = self.encoder(x)  # shape: (B, C, H, W)
        z = self.flatten(z)  # shape: (B, C*H*W)
        z = self.fc_encode(z)  # shape: (B, latent_dim)
        return z

    def decode(self, z):
        """
        潜在ベクトル z を Reshape して特徴マップに戻し、画像を復元
        """
        z = self.fc_decode(z)  # shape: (B, C*H*W)
        z = z.view(-1, *self.latent_shape)  # shape: (B, C, H, W)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        """
        画像を Flatten した潜在ベクトルに圧縮し、デコードして復元
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


if __name__ == "__main__":
    # テスト用：ダミー入力（例: 1サンプル, チャンネル3, 256x256画像）
    model = MaxVITAE('../../config/MaxVIT.yaml', latent_dim=512)
    dummy_input = torch.randn(1, 3, 256, 256)
    x_recon, z = model(dummy_input)
    print("Reconstructed output shape:", x_recon.shape)  # (B, 3, 256, 256)
    print("Latent vector shape:", z.shape)  # (B, 512)
