import sys
sys.path.append("../")

import hydra
import torch

from omegaconf import OmegaConf, DictConfig
from agents.agents import get_agents
from buffers.buffers import get_buffers
from envs.envs import make_env
from models.AutoEncoder.encoder import build_encoder
from utils.helper import encode_img, decode_img
from utils.visualize import visualize_and_save_reconstruction
from utils.timers import Timer as Timer
# from utils.timers import TimerDummy as Timer

@hydra.main(config_path='../config', config_name='sample', version_base='1.2')
def main(config: DictConfig):

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 環境の作成
    env = make_env(config.env)
    dim_info = env.get_info()
    state_dim = dim_info["info_dim"]
    action_dim = dim_info["action_space"][0]
    latent_dim = config.latent_dim

    # エージェントの作成
    agent = get_agents(config.agent, latent_dim=latent_dim, state_dim=state_dim, action_dim=action_dim, device=device)

    # バッファの作成
    buffer = get_buffers(config.buffer, latent_dim=latent_dim, state_dim=state_dim, action_dim=action_dim)

    # エンコーダの作成
    encoder = build_encoder(config.encoder, latent_dim, device)

    # 環境の初期化
    obs, vehicle_info = env.reset()
    
    max_steps = config.max_steps
    save_interval = config.get("save_interval", 10)  # デフォルト値 10

    for step in range(max_steps):  
        with Timer(f"Step {step}"):
            state_img = obs["image"].copy()  # 画像データのコピーを取得
            state = obs["vehicle"]

            with Timer("Encoding"):
                z = encode_img(encoder, state_img) # z: numpy.ndarray

            with Timer("Decoding"):
                state_recon = decode_img(encoder, z) # state_recon: numpy.ndarray

            # 画像の保存を save_interval ごとに実行
            if step % save_interval == 0:
                with Timer("Visualization"):
                    visualize_and_save_reconstruction(original=state_img,
                                                      reconstructed=state_recon,
                                                      output_dir=config.output_dir,
                                                      step=step)

            with Timer("Agent Action"):
                action = agent.select_action(z=z, state=state, evaluate=False)

            with Timer("Environment Step"):
                next_obs, reward, terminated, truncated, info = env.step(action)

            next_state_img = next_obs["image"].copy()  # 画像データのコピーを取得
            next_state = next_obs["vehicle"]

            with Timer("Next Encoding"):
                next_z = encode_img(encoder, next_state_img)  # z: numpy.ndarray

            with Timer("Buffer Add"):
                buffer.add(z, state, action, reward, next_z, next_state, terminated)

            # 状態の更新
            obs = next_obs  # ここで最新の観測を保存

            if terminated:
                with Timer("Environment Reset"):
                    obs, vehicle_info = env.reset()


if __name__ == '__main__':
    main()