from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm

class BayesianOptimizer:
    """Process parameter optimizer using Bayesian Optimization."""
    
    def __init__(
        self,
        model: Callable,
        config: dict
    ):
        """
        Initialize the optimizer.
        
        Args:
            model: Trained model that predicts the target variable
            config: OptimizationConfig instance containing optimization parameters
        """
        self.model = model
        self.config = config
    
    def _optimize_single_point(self, args: tuple) -> Dict:
        """
        Optimize parameters for a single data point.
        
        Args:
            args: Tuple of (index, data_point)
            
        Returns:
            Dictionary containing optimization results
        """
        idx, data_point = args
        
        def objective(**params):
            modified_state = data_point.copy()
            for param_name, value in params.items():
                modified_state[param_name] = value
            return -self.model.predict([modified_state])[0]
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=self.config.param_bounds
        )
        
        optimizer.maximize(
            init_points=self.config.init_points,
            n_iter=self.config.n_iter
        )
        
        return optimizer.max
    
    def optimize_batch(
        self,
        data: pd.DataFrame,
        progress_bar: bool = True
    ) -> pd.DataFrame:
        """
        Optimize parameters for multiple data points in parallel.
        
        Args:
            data: DataFrame containing input data
            progress_bar: Whether to show progress bar
            
        Returns:
            DataFrame containing optimization results for each data point
        """
        args = list(enumerate(data.iloc))
        
        with Pool(self.config.n_processes) as pool:
            if progress_bar:
                results = list(tqdm(
                    pool.imap(self._optimize_single_point, args),
                    total=len(data),
                    desc="Optimizing parameters"
                ))
            else:
                results = pool.map(self._optimize_single_point, args)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'data_index': i,
                'objective_value': -r['target'],
                **r['params']
            } for i, r in enumerate(results)
        ])
        
        return results_df