import gymnasium as gym
import numpy as np
import cv2

# 画像サイズ変更用ラッパー
class ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(128, 128)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.shape[0], self.shape[1], 3),
            dtype=np.uint8
        )

    def observation(self, obs):
        resized_obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return resized_obs

# カスタム環境
class CarRacingWithInfo(gym.Wrapper):
    def __init__(self, env_name="CarRacing-v3", render_mode=None, img_size=(128, 128)):
        env = gym.make(env_name, render_mode=render_mode)
        if img_size is not None:
            env = ResizeObservationWrapper(env, shape=img_size)
        super().__init__(env)

        # 画像サイズを取得
        self.img_shape = self.observation_space.shape
        # 数値情報の次元 (速度, 角度, 位置 (x, y))
        self.info_dim = 4

    def _get_info(self):
        car = self.unwrapped.car
        return {
            "speed": np.linalg.norm(car.hull.linearVelocity),
            "angle": car.hull.angle,
            "position": np.array(car.hull.position)
        }

    def _get_observation(self, obs):
        info = self._get_info()
        return {
            "image": (obs.astype(np.float32) / 255.0),  # 画像の正規化
            "vehicle": np.array([info["speed"], info["angle"], *info["position"]], dtype=np.float32)  # 数値情報
        }

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return self._get_observation(obs), reward, terminated, truncated, self._get_info()

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return self._get_observation(obs), self._get_info()

    def get_info(self):
        """環境の情報を取得（次元確認用）"""
        return {
            "image_shape": self.img_shape,
            "info_dim": self.info_dim,
            "action_space": self.action_space.shape if isinstance(self.action_space, gym.spaces.Box) else self.action_space.n
        }
