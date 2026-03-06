import sys
import os
# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Optimization.models.lp_model import LPOptimizationModel
from Optimization.utils.data_utils import generate_sample_demand
import numpy as np

def main():
    # Configuration
    config = {
        "capacity": 100,
        "horizon": 10,
        "min_price": 50.0,
        "max_price": 500.0,
        "num_breakpoints": 11,
        "use_soft_capacity": False
    }

    print(f"--- Initializing {config['horizon']}-day RevPAR LP Optimization ---")
    
    # Instantiate Model
    lp_model = LPOptimizationModel(config)
    
    # Generate Sample Demand
    demand_matrix = generate_sample_demand(
        horizon=config['horizon'],
        price_grid=lp_model.price_grid,
        base_demand=15.0,
        elasticity=2.0
    )
    
    lp_model.set_demand_data(demand_matrix)
    
    # Solve
    print("Solving model...")
    results = lp_model.solve()
    
    if results["status"] == "OPTIMAL":
        print("\n--- Optimization Results ---")
        print(f"Objective Value (Total Revenue): ${results['objective']:.2f}")
        print(f"Total Demand: {results['total_demand']:.2f} / {config['capacity']}")
        print("\nDaily Breakdown:")
        print("Day | Price  | Demand | Revenue")
        print("-" * 30)
        for d in range(config['horizon']):
            print(f"{d:3d} | ${results['prices'][d]:6.2f} | {results['quantities'][d]:6.2f} | ${results['revenues'][d]:7.2f}")
    else:
        print(f"Optimization failed with status: {results['status']}")

if __name__ == "__main__":
    main()
