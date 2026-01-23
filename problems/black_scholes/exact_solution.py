"""European Call Option exact solution (Black-Scholes formula)"""
import torch
from . import config


class EuropeanCallOption:
    """
    European Call Option with exact Black-Scholes pricing.
    
    Creates a grid of (time, spot) points and computes:
    - Option prices using Black-Scholes formula
    - Greeks (delta, gamma) for data filtering
    """
    
    def __init__(self, K=None, T=None, r=None, sigma=None, q=None, M=None, N=None):
        """
        Initialize European Call Option.
        
        Args:
            K: Strike price (default from config)
            T: Time to expiration (default from config)
            r: Risk-free rate (default from config)
            sigma: Volatility (default from config)
            q: Dividend yield (default from config)
            M: Number of spot price steps (default from config)
            N: Number of time steps (default from config)
        """
        # Use config values if not provided
        self.K = torch.as_tensor(K if K is not None else config.K, dtype=torch.float32)
        self.T = torch.as_tensor(T if T is not None else config.T, dtype=torch.float32)
        self.r = torch.as_tensor(r if r is not None else config.r, dtype=torch.float32)
        self.sigma = torch.as_tensor(sigma if sigma is not None else config.sigma, dtype=torch.float32)
        self.q = torch.as_tensor(q if q is not None else config.q, dtype=torch.float32)
        
        self.M = M if M is not None else config.M
        self.N = N if N is not None else config.N
        
        # Grid setup
        self.s_max = 2 * self.K
        self.dt = self.T / self.N
        self.ds = self.s_max / self.M
        
        # Create spot and time grids
        self.S = torch.linspace(0.0, self.s_max, self.M + 1, dtype=torch.float32)
        self.t = torch.linspace(0.0, self.T, self.N + 1, dtype=torch.float32)
        
        # Create meshgrid of (time, spot) points
        time_grid, spot_grid = torch.meshgrid(self.t, self.S, indexing="ij")
        self.grid_points = torch.stack((time_grid, spot_grid), dim=2)
        
        # Compute option prices on grid
        self.C = torch.zeros((N + 1, M + 1))
        
        # Boundary condition: C(0, t) = 0 (worthless if S=0)
        self.C[:, 0] = torch.as_tensor(0, dtype=torch.float32)
        
        # Terminal condition: C(S, T) = max(S - K, 0)
        self.C[-1, :] = torch.maximum(self.S - self.K, torch.tensor(0.0))
        
        # Compute prices for interior points
        reduced_grid = self.grid_points[:-1, 1:, :]
        curr_time_points, curr_spot_points = reduced_grid[..., 0], reduced_grid[..., 1]
        self.C[:-1, 1:] = self.price_option(curr_spot_points, curr_time_points)
        
        # Compute Greeks for data filtering
        reduced_grid = self.grid_points[:-1, :, :]
        curr_time_points, curr_spot_points = reduced_grid[..., 0], reduced_grid[..., 1]
        
        self.gamma = torch.zeros((N, M + 1))
        self.gamma = self.calc_gamma_option(curr_spot_points, curr_time_points)
        self.gamma[:, 0] = torch.as_tensor(0, dtype=torch.float32)
        
        self.delta = torch.zeros_like(self.gamma)
        self.delta = self.calc_delta_option(curr_spot_points, curr_time_points)
    
    def price_option(self, S, t):
        """
        Compute European Call option price using Black-Scholes formula.
        
        Args:
            S: Spot price(s)
            t: Time(s)
        
        Returns:
            Option price(s)
        """
        tau = torch.as_tensor(self.T - t, dtype=torch.float64)
        sqrt_tau = torch.sqrt(tau)
        
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        d2 = d1 - self.sigma * sqrt_tau
        
        norm = torch.distributions.Normal(0, 1)
        
        price = (S * torch.exp(-self.q * tau) * norm.cdf(d1) - 
                self.K * torch.exp(-self.r * tau) * norm.cdf(d2))
        
        return price
    
    def calc_gamma_option(self, S, t):
        """
        Compute Gamma (second derivative w.r.t. S).
        
        Args:
            S: Spot price(s)
            t: Time(s)
        
        Returns:
            Gamma value(s)
        """
        tau = self.T - t
        sqrt_tau = torch.sqrt(tau)
        
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        
        norm = torch.distributions.Normal(0, 1)
        
        gamma = (torch.exp(-self.q * tau) * norm.log_prob(d1).exp() / 
                (S * self.sigma * sqrt_tau))
        
        return gamma
    
    def calc_delta_option(self, S, t):
        """
        Compute Delta (first derivative w.r.t. S).
        
        Args:
            S: Spot price(s)
            t: Time(s)
        
        Returns:
            Delta value(s)
        """
        tau = self.T - t
        sqrt_tau = torch.sqrt(tau)
        
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        
        norm = torch.distributions.Normal(0, 1)
        
        delta = torch.exp(-self.q * tau) * norm.cdf(d1)
        
        return delta
    
    def filter_data_by_gamma(self, lower_gamma_bound, upper_gamma_bound):
        """
        Filter grid points by gamma range.
        
        Args:
            lower_gamma_bound: Minimum gamma value
            upper_gamma_bound: Maximum gamma value
        
        Returns:
            Tuple of (filtered_points, prices, gammas)
        """
        mask = (lower_gamma_bound <= self.gamma) & (self.gamma <= upper_gamma_bound)
        reduced_grid = self.grid_points[:-1, :, :]
        
        return (reduced_grid[mask], 
                self.C[:-1, :][mask].reshape(-1, 1), 
                self.gamma[mask].reshape(-1, 1))
    
    def filter_data_by_delta(self, lower_delta_bound, upper_delta_bound):
        """
        Filter grid points by delta range.
        
        Useful for focusing training on at-the-money options.
        
        Args:
            lower_delta_bound: Minimum delta value
            upper_delta_bound: Maximum delta value
        
        Returns:
            Tuple of (filtered_points, prices, gammas)
        """
        mask = (lower_delta_bound <= self.delta) & (self.delta <= upper_delta_bound)
        reduced_grid = self.grid_points[:-1, :, :]
        
        return (reduced_grid[mask], 
                self.C[:-1, :][mask].reshape(-1, 1), 
                self.gamma[mask].reshape(-1, 1))
