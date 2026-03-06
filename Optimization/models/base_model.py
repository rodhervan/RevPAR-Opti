import numpy as np
from typing import Dict, List, Any

class BaseOptimizationModel:
    """
    Base class for RevPAR optimization models.
    Provides common structures for handling capacity, pricing grids, and data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.capacity = config.get("capacity", 100)
        self.horizon = config.get("horizon", 60)
        self.min_price = config.get("min_price", 50.0)
        self.max_price = config.get("max_price", 500.0)
        self.num_breakpoints = config.get("num_breakpoints", 11)
        
        # Price grid (K)
        self.price_grid = np.linspace(self.min_price, self.max_price, self.num_breakpoints)
        
        # Dates (D)
        self.dates = list(range(self.horizon))
        
        # Performance/Logging
        self.model_name = "BaseModel"

    def set_demand_data(self, demand_matrix: np.ndarray):
        """
        Sets the predicted demand matrix D_{d,k}.
        Shape: (horizon, num_breakpoints)
        """
        assert demand_matrix.shape == (self.horizon, self.num_breakpoints)
        self.demand_matrix = demand_matrix
        self.revenue_matrix = demand_matrix * self.price_grid[np.newaxis, :]

    def solve(self):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement solve()")
