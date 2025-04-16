import gymnasium as gym
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import glob
import humanoid_envs
from gymnasium.envs.registration import register
import humanoid_envs.v0

register(
    id="v0",
    entry_point="humanoid_envs.v0:HumanoidEnv",
)

CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FREQ = 10_000
TOTAL_TIMESTEPS = 30_000
TRAIN_ENV = "v0" # Humanoid-v5, v0
MODEL_PATH = f"{TRAIN_ENV}_humanoid_ppo"
STEP_COUNT_FILE = f"{TRAIN_ENV}_step_count.txt"

env = gym.make(TRAIN_ENV, render_mode="human")

# Load latest checkpoint if it exists
def get_latest_checkpoint():
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/{TRAIN_ENV}_*_steps.zip")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

# Load previous model if it exists
if os.path.exists(MODEL_PATH + ".zip"):
    print("Loading finished model...")
    model = PPO.load(MODEL_PATH, env=env)
elif (latest_ckpt := get_latest_checkpoint()) is not None:
    print("Loading from latest checkpoint...")
    model = PPO.load(latest_ckpt, env=env)
else:
    print("Training new model...")
    model = PPO("MlpPolicy", env, verbose=1)

# Tracking step count
if os.path.exists(STEP_COUNT_FILE):
    with open(STEP_COUNT_FILE, "r") as file:
        step_count = int(file.read())
else:
    step_count = 0
    with open(STEP_COUNT_FILE, "w") as file:
        file.write(str(step_count))

remaining_timesteps = TOTAL_TIMESTEPS - step_count
if remaining_timesteps <= 0:
    print("Training finished. Running...")
else:
    print(f"Continuing training for {remaining_timesteps} more steps.")

# Custom callback to save with step count in name
class CustomCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            steps = step_count + self.n_calls
            save_path = os.path.join(self.save_path, f"{TRAIN_ENV}_{steps}_steps.zip")
            self.model.save(save_path)
        return True

checkpoint_callback = CustomCheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix=f"{TRAIN_ENV}"
)

try:
    model.learn(total_timesteps=remaining_timesteps, callback=checkpoint_callback, progress_bar=True)
finally:
    step_count += remaining_timesteps
    with open(STEP_COUNT_FILE, "w") as file:
        file.write(str(step_count))
    model.save(MODEL_PATH)

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()