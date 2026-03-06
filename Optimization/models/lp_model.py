import gurobipy as gp
from gurobipy import GRB
import numpy as np
from Optimization.models.base_model import BaseOptimizationModel

class LPOptimizationModel(BaseOptimizationModel):
    """
    Linear Programming implementation of RevPAR optimization.
    Uses a convex-combination (lambda) formulation.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "LP_ConvexCombination"
        self.softcap_penalty = config.get("softcap_penalty", 1000.0)
        self.use_soft_capacity = config.get("use_soft_capacity", False)

    def solve(self):
        # Create Gurobi Model
        m = gp.Model(self.model_name)
        
        # Decision Variables: lambda_{d,k}
        # Continuous weights for each breakpoint k on day d
        lambdas = m.addVars(self.horizon, self.num_breakpoints, lb=0.0, ub=1.0, name="lambda")
        
        # Auxiliary variables for linkage (optional but good for clarity/reporting)
        prices = m.addVars(self.horizon, lb=self.min_price, ub=self.max_price, name="price")
        quantities = m.addVars(self.horizon, lb=0.0, name="quantity")
        revenues = m.addVars(self.horizon, lb=0.0, name="revenue")
        
        # 1. Convex Combination (Sum to 1)
        for d in self.dates:
            m.addConstr(gp.quicksum(lambdas[d, k] for k in range(self.num_breakpoints)) == 1.0, name=f"sum_to_one_{d}")
            
            # 2. Variable Linkage
            m.addConstr(prices[d] == gp.quicksum(lambdas[d, k] * self.price_grid[k] for k in range(self.num_breakpoints)), name=f"link_price_{d}")
            m.addConstr(quantities[d] == gp.quicksum(lambdas[d, k] * self.demand_matrix[d, k] for k in range(self.num_breakpoints)), name=f"link_qty_{d}")
            m.addConstr(revenues[d] == gp.quicksum(lambdas[d, k] * self.revenue_matrix[d, k] for k in range(self.num_breakpoints)), name=f"link_rev_{d}")

        # 3. Capacity Constraint
        total_quantity = gp.quicksum(quantities[d] for d in self.dates)
        if self.use_soft_capacity:
            penalty_var = m.addVar(lb=0.0, name="cap_violation")
            m.addConstr(total_quantity <= self.capacity + penalty_var, name="capacity_limit")
            obj = gp.quicksum(revenues[d] for d in self.dates) - self.softcap_penalty * penalty_var
        else:
            m.addConstr(total_quantity <= self.capacity, name="capacity_limit")
            obj = gp.quicksum(revenues[d] for d in self.dates)

        # Set Objective
        m.setObjective(obj, GRB.MAXIMIZE)
        
        # Optimize
        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            results = {
                "status": "OPTIMAL",
                "objective": m.ObjVal,
                "prices": [prices[d].X for d in self.dates],
                "quantities": [quantities[d].X for d in self.dates],
                "revenues": [revenues[d].X for d in self.dates],
                "total_demand": sum(quantities[d].X for d in self.dates)
            }
            return results
        else:
            return {"status": "NOT_OPTIMAL", "code": m.Status}
