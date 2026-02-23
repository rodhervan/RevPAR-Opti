import os
import warnings
import logging

# Suppress Ray deprecation and environment overriding warnings
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from hotel_pricing_env import HotelPricingEnv
from ray.tune.logger import NoopLogger
import torch

def env_creator(env_config):
    return HotelPricingEnv(env_config)

# 1. Register Environment
register_env("HotelPricingEnv-v1", env_creator)

def custom_logger_creator(config):
    # Fixes the UnifiedLogger / JsonLogger / CSVLogger / TBXLogger deprecations
    return NoopLogger(config, os.path.abspath("./rllib_logs"))

def main():
    # 2. Initialize Ray safely
    ray.init(ignore_reinit_error=True)
    
    # Base Configuration
    env_config = {
        "capacity": 100, 
        "horizon": 60,
        "min_price": 50.0,
        "max_price": 400.0,
        "action_type": "continuous" # Switch to 'discrete' to test grid behavior
    }
    
    # 3. Configure PPO Agent
    config = (
        PPOConfig()
        .environment(
            env="HotelPricingEnv-v1",
            env_config=env_config
        )
        .framework("torch")
        # Fixes multi_gpu_train_one_step deprecation by enabling natively the New API Stack
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(num_env_runners=3)
        .debugging(logger_creator=custom_logger_creator)
        .training(
            gamma=0.99,            # Discount factor (values near 1 handle delayed sequences better)
            lr=5e-5,               # Learning rate
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,         # Fixes num_sgd_iter deprecation
            vf_clip_param=5000.0   # Prevent value function clipping warnings
        )
    )

    # 4. Build Algorithm
    algo = config.build_algo() # Fixes .build() deprecation
    
    # Setup checkpoint dir
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints_rl"))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 5. Training Loop
    num_iterations = 20
    print(f"Starting Training for {num_iterations} iterations...")
    
    for i in range(num_iterations):
        result = algo.train()
        
        env_runners_stats = result.get('env_runners', {})
        mean_reward = env_runners_stats.get('episode_return_mean', 
                      env_runners_stats.get('episode_reward_mean', 0.0))
        episode_len = env_runners_stats.get('episode_len_mean', 0.0)
                      
        print(f"Iter {i:02d} | Mean Rev (Reward): ${mean_reward:>7.2f} | Ep Length: {episode_len:>5.1f}")
        
    # Save final model
    save_result = algo.save(checkpoint_dir=ckpt_dir)
    print(f"Final Model Saved to: {save_result.checkpoint.path}")
    
    ray.shutdown()

if __name__ == "__main__":
    main()