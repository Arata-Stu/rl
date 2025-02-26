import sys
sys.path.append("../")

import hydra
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # TensorBoard用

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

    # TensorBoardの初期化
    writer = SummaryWriter(log_dir=config.get("log_dir", "./runs"))

    # 環境の作成
    env = make_env(config.envs)
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
    encoder.to(device)

    # 環境の初期化
    obs, vehicle_info = env.reset()

    # 各種パラメータの取得
    max_steps = config.total_steps
    save_imgs = config.get("save_imgs", False)          # 画像保存の有無
    save_interval = config.get("save_interval", 10)       # 画像保存の間隔
    train_start_steps = config.get("train_start_steps", 10000)  # 学習開始までのウォームアップステップ数
    batch_size = config.get("batch_size", 64)
    train_interval = config.get("train_interval", 4)      # 何ステップごとに学習するか
    total_episodes = config.get("total_episodes", 1000)
    save_model_interval = config.get("save_model_interval", 10)

    episode_rewards = []
    top_models = []

    try:
        # エピソードループ
        for episode in range(total_episodes):
            obs, vehicle_info = env.reset()
            episode_reward = 0

            # ステップループ
            for step in range(max_steps):
                with Timer(f"Step {step}"):
                    state_img = obs["image"].copy()
                    state = obs["vehicle"]
                    # print(f"speed: {state[0]}")

                    with Timer("Encoding"):
                        z = encode_img(encoder, state_img)

                    # with Timer("Decoding"):
                    #     state_recon = decode_img(encoder, z)

                    # if save_imgs and step % save_interval == 0:
                    #     with Timer("Visualization"):
                    #         visualize_and_save_reconstruction(original=state_img,
                    #                                           reconstructed=state_recon,
                    #                                           output_dir=config.output_dir,
                    #                                           step=step)

                    with Timer("Agent Action"):
                        action = agent.select_action(z=z, state=state, evaluate=False, action_space=env.action_space)
                        # print(f"Action: {action}")

                    # TensorBoard にアクションの分布を記録
                    writer.add_histogram("Action/Distribution", action, episode)

                    with Timer("Environment Step"):
                        next_obs, reward, terminated, truncated, info = env.step(action)

                    next_state_img = next_obs["image"].copy()
                    next_state = next_obs["vehicle"]

                    with Timer("Next Encoding"):
                        next_z = encode_img(encoder, next_state_img)

                    with Timer("Buffer Add"):
                        buffer.add(z, state, action, reward, next_z, next_state, terminated)

                    episode_reward += reward

                    # 学習開始の条件チェック
                    if len(buffer) > train_start_steps and step % train_interval == 0:
                        with Timer("Training"):
                            batch = buffer.sample(batch_size)
                            loss = agent.update(batch)

                            for loss_name, loss_value in loss.items():
                                writer.add_scalar(f"Loss/Training/{loss_name}", loss_value, episode * max_steps + step)


                    obs = next_obs

                    if terminated or truncated:
                        with Timer("Environment Reset"):
                            obs, vehicle_info = env.reset()
                            print(f"Episode {episode}: Step {step} terminated because of terminated: {terminated} or truncated: {truncated}")
                        break

            episode_rewards.append(episode_reward)
            writer.add_scalar("Reward/Episode", episode_reward, episode)

            # トップモデルの保存処理
            if len(top_models) < 3:
                top_models.append((episode, episode_reward))
                agent.save(f"{config.model_dir}/best_{episode_reward:.2f}_ep_{episode}.pt", episode)
            else:
                min_reward = min(top_models, key=lambda x: x[1])[1]
                if episode_reward > min_reward:
                    top_models = [model for model in top_models if model[1] != min_reward]
                    top_models.append((episode, episode_reward))
                    agent.save(f"{config.model_dir}/best_{episode_reward:.2f}_ep_{episode}.pt", episode)
            # 降順ソート
            top_models = sorted(top_models, key=lambda x: x[1], reverse=True)

            if episode % save_model_interval == 0:
                agent.save(f"{config.model_dir}/regular_ep_{episode}.pt", episode)

            print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 最後に報酬を保存し、TensorBoardのwriterをクローズする
        # np.save(f"{config.output_dir}/episode_rewards.npy", np.array(episode_rewards))
        writer.close()
        print("Cleaned up resources.")

if __name__ == '__main__':
    main()
