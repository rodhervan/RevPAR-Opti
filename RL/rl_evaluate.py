import os
import warnings
import logging

# Suppress Ray deprecation and environment overriding warnings
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

import ray
import numpy as np
import matplotlib.pyplot as plt
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from hotel_pricing_env import HotelPricingEnv
from ray.tune.logger import NoopLogger

def env_creator(env_config):
    return HotelPricingEnv(env_config)

register_env("HotelPricingEnv-v1", env_creator)

def custom_logger_creator(config):
    # Fixes the UnifiedLogger / JsonLogger / CSVLogger / TBXLogger deprecations natively
    return NoopLogger(config, os.path.abspath(os.path.join(os.path.dirname(__file__), "rllib_logs_eval")))

def evaluate_agent(checkpoint_path: str):
    ray.init(ignore_reinit_error=True)
    
    # Load the trained algorithm
    try:
        # Pass custom logger to avoid deprecation warnings when loading
        algo = Algorithm.from_checkpoint(
            checkpoint_path,
        )
        print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    env_config = {
        "capacity": 100, 
        "horizon": 60,
        "min_price": 50.0,
        "max_price": 400.0,
        "action_type": "continuous"
    }

    env = HotelPricingEnv(env_config)
    obs, info = env.reset()

    # Using New API Stack natively for inference
    module = algo.get_module("default_policy")

    prices = []
    revenues = []
    capacities = []
    time_steps = []
    demands = []

    done = False
    total_reward = 0.0

    print("\n--- Starting Evaluation Episode ---")
    while not done:
        # Convert observation to discrete tensor batch [1, obs_size]
        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32)
        
        # Native fix for compute_single_action deprecation warning
        with torch.no_grad():
            out = module.forward_inference({"obs": obs_tensor})
            
        # Extract scalar action from PyTorch output
        action_dist = out.get("action_dist_inputs")
        action = action_dist[0].numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        prices.append(info["price_set"])
        revenues.append(info["revenue"])
        capacities.append(info["remaining_capacity"])
        demands.append(info["realized_demand"])
        time_steps.append(info["time_to_stay"])
        
        done = terminated or truncated

    print(f"Episode Finished! Total Revenue Generated: ${total_reward:.2f}")
    print(f"Final Capacity Remaining: {capacities[-1]}")
    
    # Optional: Plot the results to visualize the agent's strategy
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    # Reverse time steps to show time moving forward towards stay date
    x_axis = np.arange(len(time_steps))
    plt.plot(x_axis, prices, marker='o', color='blue', label="Price Set")
    plt.axhline(env_config["min_price"], color='r', linestyle='--', alpha=0.5, label='Min Price')
    plt.axhline(env_config["max_price"], color='r', linestyle='--', alpha=0.5, label='Max Price')
    plt.ylabel("Price ($)")
    plt.title("Trained RL Agent Pricing Strategy Over Booking Horizon")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.step(x_axis, capacities, where='post', color='green', label="Remaining Capacity")
    plt.ylabel("Rooms Available")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.bar(x_axis, demands, alpha=0.6, color='purple', label="Realized Demand")
    plt.bar(x_axis, [r/max(1, p) for r, p in zip(revenues, prices)], alpha=0.6, color='orange', label="Actual Sales")
    plt.ylabel("Rooms")
    plt.xlabel("Days Since Start of Booking Window")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "rl_evaluation_plot.png")
    plt.savefig(plot_path)
    print(f"Saved evaluation plot to {plot_path}")
    
    ray.shutdown()

if __name__ == "__main__":
    import os
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints_rl")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint directory not found at {ckpt_path}")
    else:
        evaluate_agent(ckpt_path)
