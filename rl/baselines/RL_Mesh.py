import os
from pathlib import Path
import torch
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from rl.boundary_env import BoudaryEnv, read_polygon
from rl.baselines.CustomizeCallback import CustomCallback
import warnings
import multiprocessing as mp
from tqdm import tqdm
import time

def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

class TqdmCallback(BaseCallback):
    """
    Custom callback for displaying training progress with tqdm
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.start_time = None
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            unit="steps",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
            
            # Update description with current stats every 1000 steps
            if self.num_timesteps % 1000 == 0:
                elapsed_time = time.time() - self.start_time
                steps_per_sec = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                
                # Get recent reward if available
                recent_reward = "N/A"
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    recent_reward = f"{self.model.ep_info_buffer[-1]['r']:.2f}"
                
                self.pbar.set_description(
                    f"Training | Steps/sec: {steps_per_sec:.1f} | Recent Reward: {recent_reward}"
                )
        return True
    
    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

def train_model(method_name, train_env, eval_env, total_timesteps, log_path, tensorboard_log, seed):
    """
    Trains a new reinforcement learning model from scratch.

    :param method_name: The RL algorithm to use (e.g., 'sac', 'ppo').
    :param train_env: The environment for training.
    :param eval_env: The environment for evaluation during training.
    :param total_timesteps: The total number of steps to train for.
    :param log_path: The directory to save model checkpoints and logs.
    :param tensorboard_log: The directory to save TensorBoard logs.
    :param seed: The random seed for reproducibility.
    """
    # 1. Create the log directories
    mkdir_p(log_path)
    mkdir_p(tensorboard_log)

    # 2. Set up the evaluation callback
    # This will periodically evaluate the model and save the best one.
    eval_callback = CustomCallback(eval_env, best_model_save_path=log_path,
                                   log_path=log_path, eval_freq=5000,
                                   n_eval_episodes=5,
                                   deterministic=True, render=False)

    # 3. Configure and instantiate the RL model  
    # Note: RTX 5090 has compatibility issues with current PyTorch, using CPU
    device = torch.device("cuda")
    
    model_params = {
        'sac': lambda: SAC(
            'MlpPolicy', train_env, seed=seed, 
            policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128, 128]),
            learning_rate=3e-4, learning_starts=10000, batch_size=100,
            tensorboard_log=tensorboard_log, device=device
        ),
        'ppo': lambda: PPO(
            'MlpPolicy', train_env, seed=seed,
            policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
            tensorboard_log=tensorboard_log, device=device
        ),
        # Add other model configurations here if needed
    }

    if method_name not in model_params:
        raise ValueError(f"Unknown RL method: {method_name}")
    
    model = model_params[method_name]()

    # 4. Set up progress tracking callback
    progress_callback = TqdmCallback(total_timesteps)
    
    # 5. Start the training process
    print(f"Starting training with {method_name.upper()} for {total_timesteps} timesteps...")
    print(f"Device: {device}")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[eval_callback, progress_callback]
    )

    # 6. Save the final model
    final_model_path = f"{log_path}/final_model.zip"
    model.save(final_model_path)
    print(f"--- Training Complete ---")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved in: {log_path}")
    print(f"To view logs, run: tensorboard --logdir {tensorboard_log}")
    print("-------------------------")


if __name__ == "__main__":
    # --- GPU-Optimized Configuration ---
    # You can change these parameters to customize the training run.
    RL_METHOD = 'sac'  # Algorithm: 'sac' or 'ppo' (SAC generally better for continuous control)
    TOTAL_TIMESTEPS = 2000000  # Increased for better GPU utilization and results
    SEED = 42  # Random seed for consistent results.
    VERSION = "basic1_gpu_training"  # A name for this training run, used for the log folder.

    # --- Path Setup ---
    # This section automatically sets up the necessary paths.
    base_path = Path(__file__).parent.parent.parent
    domain_path = base_path / "ui" / "domains" / "basic1.json"
    log_path = base_path / "logs" / f"{RL_METHOD}_{VERSION}"
    tensorboard_log_path = base_path / "tensorboard_logs" / f"{RL_METHOD}_{VERSION}"

    # --- Execution ---
    if not domain_path.exists():
        print(f"Error: Domain file not found at {domain_path}")
    else:
        # 1. Set up the environments
        print("Setting up training and evaluation environments...")
        train_env = BoudaryEnv(read_polygon(domain_path))
        eval_env = BoudaryEnv(read_polygon(domain_path))

        # 2. Start the training process
        train_model(
            method_name=RL_METHOD,
            train_env=train_env,
            eval_env=eval_env,
            total_timesteps=TOTAL_TIMESTEPS,
            log_path=str(log_path),
            tensorboard_log=str(tensorboard_log_path),
            seed=SEED
        )
