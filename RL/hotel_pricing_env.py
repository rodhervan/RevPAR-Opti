import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

class HotelPricingEnv(gym.Env):
    """
    Custom Environment for dynamic hotel revenue optimization that follows gymnasium interface.
    Implements the finite-horizon MDP with continuous or discrete pricing options.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        
        # Extract environment parameters
        self.capacity = env_config.get("capacity", 100)
        self.horizon = env_config.get("horizon", 60) # Days before stay
        
        # Pricing Bounds
        self.min_price = env_config.get("min_price", 50.0)
        self.max_price = env_config.get("max_price", 500.0)
        
        # Action space configuration
        self.action_type = env_config.get("action_type", "continuous")
        if self.action_type == "discrete":
            self.num_prices = env_config.get("num_prices", 10)
            self.price_grid = np.linspace(self.min_price, self.max_price, self.num_prices)
            self.action_space = spaces.Discrete(self.num_prices)
        else:
            # Continuous action space between -1 and 1 (standard for PPO)
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(1,), 
                dtype=np.float32
            )
            
        # Contextual features X_t
        # Feature 1: Base Demand Index (used heavily in the PE model)
        self.num_context_features = 1
        
        # State:
        # [ Remaining capacity fraction (C_t / C), Time remaining fraction (t / T), Context Features... ]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0] + [0.0] * self.num_context_features, dtype=np.float32),
            high=np.array([1.0, 1.0] + [1.0] * self.num_context_features, dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self.current_capacity = self.capacity
        self.current_time = self.horizon
        self.current_context = np.zeros(self.num_context_features, dtype=np.float32)

    def _get_price_from_action(self, action: np.ndarray | int) -> float:
        """Converts the raw action from the agent into a concrete price value."""
        if self.action_type == "discrete":
            return float(self.price_grid[int(action)])
        else:
            # Map continuous action [-1, 1] to [min_price, max_price]
            a = float(action[0])
            a = np.clip(a, -1.0, 1.0)
            price = self.min_price + 0.5 * (a + 1.0) * (self.max_price - self.min_price)
            return price

    def _demand_model(self, price: float) -> int:
        """
        D_t ~ D(p_t, X_t)
        Simulates stochastic demand based on price context.
        Uses a truncated Poisson distribution driven by a log-linear demand curve.
        """
        # Base demand scales around 5-20 rooms depending on context
        base_demand = 5.0 + (self.current_context[0] * 15.0) 
        
        # Simple elasticity assumption (higher price -> exponentially less demand)
        # We scale price relative to average for stability
        avg_price = (self.min_price + self.max_price) / 2.0
        normalized_price = price / avg_price
        elasticity = 2.0 # -2.0 price elasticity of demand
        
        expected_demand = base_demand * np.exp(-elasticity * (normalized_price - 1.0))
        
        # Realized stochastic demand
        realized_demand = self.np_random.poisson(max(0.001, expected_demand))
        return realized_demand

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_capacity = self.capacity
        self.current_time = self.horizon
        
        # Initialize context (e.g., base demand strength for this episode)
        self.current_context = self.np_random.uniform(low=0.2, high=0.8, size=(self.num_context_features,)).astype(np.float32)
        
        obs = self._get_obs()
        info = {
            "capacity": self.current_capacity,
            "time_to_stay": self.current_time
        }
        
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Action translation
        price = self._get_price_from_action(action)
        
        # 2. Demand realization
        D_t = self._demand_model(price)
        
        # 3. Actual sales (q_t = min(D_t, C_t))
        sales = min(D_t, self.current_capacity)
        
        # 4. Capacity evolution
        self.current_capacity -= sales
        
        # 5. Time evolution
        self.current_time -= 1
        
        # 6. Context evolution (Random walk with bounds)
        drift = self.np_random.normal(0, 0.05, size=self.current_context.shape)
        self.current_context = np.clip(self.current_context + drift, 0.0, 1.0).astype(np.float32)
        
        # Reward Function (R_t = p_t * q_t)
        revenue = price * sales
        
        # Termination conditions (t=0 or C_t=0)
        terminated = bool(self.current_time <= 0 or self.current_capacity <= 0)
        truncated = False
        
        obs = self._get_obs()
        info = {
            "price_set": price,
            "realized_demand": D_t,
            "sales": sales,
            "revenue": revenue,
            "remaining_capacity": self.current_capacity,
            "time_to_stay": self.current_time
        }
        
        return obs, revenue, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.current_capacity / float(self.capacity),
            self.current_time / float(self.horizon),
            *self.current_context
        ], dtype=np.float32)
