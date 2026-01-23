"""Training data generation for Black-Scholes PINN"""
import torch


class TrainingDataLoader:
    """
    Generates training data for Black-Scholes PINN.
    
    Creates three types of training points:
    1. Collocation points: Interior points for PDE loss
    2. Boundary points: S=0 for boundary condition
    3. Expiration points: t=T for terminal condition
    """
    
    def __init__(self, option, lower_delta_bound, upper_delta_bound, 
                 num_coll_points_train, num_exp_points_train, num_boundary_points_train):
        """
        Initialize training data loader.
        
        Args:
            option: EuropeanCallOption instance
            lower_delta_bound: Lower bound for delta filtering
            upper_delta_bound: Upper bound for delta filtering
            num_coll_points_train: Number of collocation points
            num_exp_points_train: Number of expiration points
            num_boundary_points_train: Number of boundary points
        """
        self.option = option
        
        self.lower_delta_bound = lower_delta_bound
        self.upper_delta_bound = upper_delta_bound
        
        self.num_coll_points_train = num_coll_points_train
        self.num_exp_points_train = num_exp_points_train
        self.num_boundary_points_train = num_boundary_points_train
        
        # Generate collocation points (filtered by delta)
        self._generate_collocation_points()
        
        # Generate expiration points (t=T)
        self._generate_expiration_points()
        
        # Generate boundary points (S=0)
        self._generate_boundary_points()
    
    def _generate_collocation_points(self):
        """Generate collocation points for physics loss"""
        # Filter points by delta (focus on at-the-money options)
        coll_points, call_prices, gamma_weights = self.option.filter_data_by_delta(
            self.lower_delta_bound, self.upper_delta_bound
        )
        
        num_coll_points = call_prices.size()[0]
        
        assert self.num_coll_points_train <= num_coll_points, \
            f"Cannot select {self.num_coll_points_train} collocation points from {num_coll_points} available points."
        
        # Randomly sample training points
        coll_shuffle = torch.randperm(num_coll_points)[:self.num_coll_points_train]
        
        self.coll_points_train = coll_points[coll_shuffle]
        self.call_prices_compare = call_prices[coll_shuffle]
        self.coll_target = torch.zeros(self.num_coll_points_train, 1)  # PDE residual should be zero
        self.coll_gamma_weights = gamma_weights[coll_shuffle]
    
    def _generate_expiration_points(self):
        """Generate expiration points for terminal condition"""
        grid_points = self.option.grid_points
        
        # Points at expiration (t=T)
        exp_points = grid_points[-1]
        call_prices_exp = self.option.C[-1, :]
        
        num_exp_points = exp_points.size()[0]
        
        assert self.num_exp_points_train <= num_exp_points, \
            f"Cannot select {self.num_exp_points_train} expiration points from {num_exp_points} available points."
        
        # Randomly sample training points
        exp_shuffle = torch.randperm(num_exp_points)[:self.num_exp_points_train]
        
        self.exp_points_train = exp_points[exp_shuffle]
        self.exp_target = call_prices_exp[exp_shuffle]
    
    def _generate_boundary_points(self):
        """Generate boundary points for S=0 condition"""
        grid_points = self.option.grid_points
        
        # Points at S=0 boundary
        boundary_points = grid_points[:, 0]
        call_prices_boundary = self.option.C[:, 0]
        
        num_boundary_points = boundary_points.size()[0]
        
        assert self.num_boundary_points_train <= num_boundary_points, \
            f"Cannot select {self.num_boundary_points_train} boundary points from {num_boundary_points} available points."
        
        # Randomly sample training points
        boundary_shuffle = torch.randperm(num_boundary_points)[:self.num_boundary_points_train]
        
        self.boundary_points_train = boundary_points[boundary_shuffle]
        self.boundary_target = call_prices_boundary[boundary_shuffle]
    
    def get_collocation_training_data(self):
        """
        Get collocation training data.
        
        Returns:
            Tuple of (points, comparison_prices, targets, gamma_weights)
        """
        return (self.coll_points_train, 
                self.call_prices_compare, 
                self.coll_target, 
                self.coll_gamma_weights)
    
    def get_expiration_training_data(self):
        """
        Get expiration training data.
        
        Returns:
            Tuple of (points, targets)
        """
        return self.exp_points_train, self.exp_target.reshape(-1, 1)
    
    def get_boundary_training_data(self):
        """
        Get boundary training data.
        
        Returns:
            Tuple of (points, targets)
        """
        return self.boundary_points_train, self.boundary_target.reshape(-1, 1)
