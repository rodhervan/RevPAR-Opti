import numpy as np

def generate_sample_demand(horizon: int, price_grid: np.ndarray, base_demand: float = 10.0, elasticity: float = 2.0) -> np.ndarray:
    """
    Generates a sample demand matrix D_{d,k} based on a log-linear model.
    Similar to the one used in the RL environment for consistency.
    """
    num_breakpoints = len(price_grid)
    demand_matrix = np.zeros((horizon, num_breakpoints))
    
    avg_price = (price_grid[0] + price_grid[-1]) / 2.0
    
    for d in range(horizon):
        # Vary base demand slightly over time (simple seasonality)
        daily_base = base_demand * (1.0 + 0.2 * np.sin(2 * np.pi * d / horizon))
        for k, p in enumerate(price_grid):
            normalized_price = p / avg_price
            expected_demand = daily_base * np.exp(-elasticity * (normalized_price - 1.0))
            demand_matrix[d, k] = expected_demand
            
    return demand_matrix
