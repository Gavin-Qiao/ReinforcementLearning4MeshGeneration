import os
import sys
import json
import argparse
import configparser
from pathlib import Path
import numpy as np
import torch
from typing import Optional

from stable_baselines3 import A2C, DDPG, SAC, PPO, TD3
from rl.boundary_env import BoudaryEnv, read_polygon, boundary
# Import RL_Mesh functions individually to avoid executing module code
try:
    from rl.baselines.RL_Mesh import mkdir_p
except ImportError:
    def mkdir_p(path):
        os.makedirs(path, exist_ok=True)
try:
    from general.EBRD import start_training, training
except ImportError:
    print("Warning: EBRD module not found, using dummy functions")
    def start_training(*args, **kwargs):
        print("Dummy start_training called")
    def training(*args, **kwargs):
        print("Dummy training called")

try:
    from general.polygon import experiement_boundaries
except ImportError:
    print("Warning: polygon module not found")
    def experiement_boundaries():
        return None

try:
    from general.mesh import MeshGeneration
except ImportError:
    print("Warning: mesh module not found")
    MeshGeneration = None


def load_config() -> configparser.ConfigParser:
    """Load configuration file."""
    base_path = Path(__file__).parent
    config = configparser.ConfigParser()
    config.read(f'{base_path}/config')
    return config


def read_boundary_data(domain_file: str):
    """Read boundary data from domain file."""
    print(f"Reading boundary data from: {domain_file}")
    try:
        boundary_data = read_polygon(domain_file)
        print(f"Successfully loaded boundary with {len(boundary_data.vertices)} vertices")
        return boundary_data
    except Exception as e:
        print(f"Error reading boundary data: {e}")
        return None


def setup_rl_environment(boundary_data, config):
    """Setup reinforcement learning environment."""
    print("Setting up RL environment...")
    try:
        env = BoudaryEnv(boundary_data)
        eval_env = BoudaryEnv(boundary_data)
        print("RL environment setup complete")
        return env, eval_env
    except Exception as e:
        print(f"Error setting up RL environment: {e}")
        return None, None


def run_rl_training(env, eval_env, config, version: str = '77', total_timesteps: int = 4000000):
    """Run reinforcement learning training."""
    print(f"Starting RL training (version: {version}, timesteps: {total_timesteps})...")
    
    try:
        # Setup RL parameters
        seed = 999
        rl_method = 'sac'
        learning_rate = 3e-4
        parameter_tuning = {'gamma': 0.5}
        
        # Create directories
        mkdir_p(f'models/{version}/')
        mkdir_p(f'plots/{version}/')
        mkdir_p(f'samples/{version}/')
        
        # Initialize and run training based on method
        if rl_method == 'sac':
            model = SAC('MlpPolicy', env, learning_rate=learning_rate, 
                       gamma=parameter_tuning['gamma'], verbose=1, seed=seed,
                       tensorboard_log=f"./runs/{version}/")
        elif rl_method == 'ppo':
            model = PPO('MlpPolicy', env, learning_rate=learning_rate,
                       gamma=parameter_tuning['gamma'], verbose=1, seed=seed,
                       tensorboard_log=f"./runs/{version}/")
        else:
            raise ValueError(f"Unsupported RL method: {rl_method}")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(save_freq=50000, 
                                               save_path=f'./models/{version}/',
                                               name_prefix='rl_model')
        
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        
        # Save final model
        model.save(f"models/{version}/final_model")
        
        print("RL training completed successfully")
        return model
        
    except Exception as e:
        print(f"Error during RL training: {e}")
        return None


def extract_training_samples(env, model, num_episodes: int = 100):
    """Extract training samples from RL model."""
    print(f"Extracting training samples from {num_episodes} episodes...")
    
    all_samples = []
    all_output_types = []
    all_outputs = []
    
    try:
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_samples = []
            episode_output_types = []
            episode_outputs = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                
                if 'samples' in info:
                    episode_samples.extend(info['samples'])
                if 'output_types' in info:
                    episode_output_types.extend(info['output_types'])
                if 'outputs' in info:
                    episode_outputs.extend(info['outputs'])
            
            all_samples.extend(episode_samples)
            all_output_types.extend(episode_output_types)
            all_outputs.extend(episode_outputs)
            
            if episode % 10 == 0:
                print(f"Processed episode {episode}/{num_episodes}")
        
        print(f"Extracted {len(all_samples)} training samples")
        return all_samples, all_output_types, all_outputs
        
    except Exception as e:
        print(f"Error extracting training samples: {e}")
        return [], [], []


def run_ann_training(samples, output_types, outputs, version: str = 'main_pipeline'):
    """Run artificial neural network training."""
    print("Starting ANN training...")
    
    try:
        # Create directories
        base_path = './'
        mkdir_p(f'{base_path}models/{version}/')
        mkdir_p(f'{base_path}plots/{version}/')
        
        model_path = f"{base_path}models/{version}/ebrd_model.pt"
        data_path = f"{base_path}samples/{version}/"
        tensorboard_log = f"./runs/{version}_ann/"
        
        # Save samples for ANN training
        sample_data = {
            'samples': samples,
            'output_types': output_types,
            'outputs': outputs
        }
        
        with open(f'{data_path}training_samples.json', 'w') as f:
            json.dump(sample_data, f)
        
        # Start ANN training
        start_training(model_path, data_path, tensorboard_log)
        
        print("ANN training completed successfully")
        return model_path
        
    except Exception as e:
        print(f"Error during ANN training: {e}")
        return None


def generate_final_mesh(env, ann_model_path: str, version: str = 'main_pipeline'):
    """Generate final mesh using trained models."""
    print("Generating final mesh...")
    
    try:
        # Load trained ANN model
        if os.path.exists(ann_model_path):
            # Run mesh generation
            training(env, version)
            print("Final mesh generation completed successfully")
            return True
        else:
            print(f"ANN model not found at: {ann_model_path}")
            return False
            
    except Exception as e:
        print(f"Error during mesh generation: {e}")
        return False


def save_results(output_dir: str, version: str):
    """Save and organize results."""
    print(f"Saving results to: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy generated meshes and plots
        import shutil
        
        source_dirs = [
            f'models/{version}/',
            f'plots/{version}/',
            f'samples/{version}/',
            f'elements/{version}/'
        ]
        
        for source_dir in source_dirs:
            if os.path.exists(source_dir):
                dest_dir = os.path.join(output_dir, os.path.basename(source_dir.rstrip('/')))
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)
                print(f"Copied {source_dir} to {dest_dir}")
        
        print("Results saved successfully")
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def main():
    """Main pipeline for RL-based mesh generation."""
    parser = argparse.ArgumentParser(description='RL-based Mesh Generation Pipeline')
    parser.add_argument('--domain', type=str, default='ui/domains/dolphine3.json',
                       help='Path to domain boundary file')
    parser.add_argument('--version', type=str, default='main_pipeline',
                       help='Version name for this run')
    parser.add_argument('--rl-timesteps', type=int, default=1000000,
                       help='Number of RL training timesteps')
    parser.add_argument('--sample-episodes', type=int, default=50,
                       help='Number of episodes for sample extraction')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip RL training (use existing model)')
    parser.add_argument('--skip-ann', action='store_true',
                       help='Skip ANN training (use existing model)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL-BASED MESH GENERATION PIPELINE")
    print("=" * 60)
    print(f"Domain file: {args.domain}")
    print(f"Version: {args.version}")
    print(f"RL timesteps: {args.rl_timesteps}")
    print(f"Sample episodes: {args.sample_episodes}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Step 1: Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config()
    
    # Step 2: Read boundary data
    print("\n[2/7] Reading boundary data...")
    boundary_data = read_boundary_data(args.domain)
    if boundary_data is None:
        print("Failed to read boundary data. Exiting.")
        return 1
    
    # Step 3: Setup RL environment
    print("\n[3/7] Setting up RL environment...")
    env, eval_env = setup_rl_environment(boundary_data, config)
    if env is None or eval_env is None:
        print("Failed to setup RL environment. Exiting.")
        return 1
    
    # Step 4: RL Training
    print("\n[4/7] RL Training...")
    if not args.skip_rl:
        rl_model = run_rl_training(env, eval_env, config, args.version, args.rl_timesteps)
        if rl_model is None:
            print("RL training failed. Exiting.")
            return 1
    else:
        print("Skipping RL training (using existing model)")
        # Try to load existing model
        try:
            rl_model = SAC.load(f"models/{args.version}/final_model")
        except:
            print("Could not load existing RL model. Exiting.")
            return 1
    
    # Step 5: Extract training samples
    print("\n[5/7] Extracting training samples...")
    samples, output_types, outputs = extract_training_samples(env, rl_model, args.sample_episodes)
    if len(samples) == 0:
        print("No training samples extracted. Exiting.")
        return 1
    
    # Step 6: ANN Training
    print("\n[6/7] ANN Training...")
    if not args.skip_ann:
        ann_model_path = run_ann_training(samples, output_types, outputs, args.version)
        if ann_model_path is None:
            print("ANN training failed. Exiting.")
            return 1
    else:
        print("Skipping ANN training (using existing model)")
        ann_model_path = f"./models/{args.version}/ebrd_model.pt"
    
    # Step 7: Generate final mesh
    print("\n[7/7] Generating final mesh...")
    success = generate_final_mesh(env, ann_model_path, args.version)
    if not success:
        print("Mesh generation failed. Exiting.")
        return 1
    
    # Save results
    print("\nSaving results...")
    save_results(args.output_dir, args.version)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
