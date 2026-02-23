import gymnasium as gym
import numpy as np
from gymnasium import spaces
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

class HotelPricingEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        
        self.max_capacity = config.get("capacity", 100) if config else 100
        self.max_dba = config.get("max_dba", 30) if config else 30
        
        self.price_grid = np.array([50, 75, 100, 125, 150, 175, 200])
        self.num_prices = len(self.price_grid)
        
        self.action_space = spaces.Discrete(self.num_prices)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.max_dba, self.max_capacity], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize internal state variables
        self.current_dba = self.max_dba
        self.remaining_capacity = self.max_capacity

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_dba = self.max_dba
        self.remaining_capacity = self.max_capacity
        
        observation = np.array([self.current_dba, self.remaining_capacity], dtype=np.float32)
        info = {}
        
        # This explicit return prevents the NoneType unpacking error
        return observation, info

    def step(self, action):
        price_t = self.price_grid[action]
        
        expected_demand = 20 - 0.05 * price_t + (self.current_dba * 0.2)
        stochastic_noise = np.random.normal(loc=0.0, scale=3.0)
        actual_demand = max(0, int(expected_demand + stochastic_noise))
        
        rooms_sold = min(actual_demand, self.remaining_capacity)
        self.remaining_capacity -= rooms_sold
        
        self.current_dba -= 1
        
        reward = price_t * rooms_sold
        
        terminated = bool(self.current_dba <= 0)
        truncated = False
        
        observation = np.array([self.current_dba, self.remaining_capacity], dtype=np.float32)
        
        info = {
            "price_set": price_t,
            "demand": actual_demand,
            "rooms_sold": rooms_sold
        }
        
        return observation, reward, terminated, truncated, info

# Initialize Ray (ignore_reinit_error prevents crashes if you rerun this cell in a notebook)
ray.init(ignore_reinit_error=True)

def env_creator(env_config):
    return HotelPricingEnv(env_config)

register_env("HotelPricingEnv-v0", env_creator)

config = (
    PPOConfig()
    .environment(
        env="HotelPricingEnv-v0",
        env_config={"capacity": 100, "max_dba": 30}
    )
    .framework("torch")
    .env_runners(num_env_runners=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    
    # Safely extract metrics accommodating the New API Stack naming conventions
    env_runners_stats = result.get('env_runners', {})
    mean_reward = env_runners_stats.get('episode_return_mean', 
                  env_runners_stats.get('episode_reward_mean', 0.0))
    
    print(f"Iteration {i}: Mean Revenue = {mean_reward:.2f}")

# ray.shutdown()