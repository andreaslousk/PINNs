import torch

class EuropeanCallOption:
    def __init__(self, K, T, r, sigma, q, M, N):

        # initialize strike (K), expiration (T), risk-free rate (r), volatility (sigma), and continuous dividends (q)
        
        self.K = torch.as_tensor(K, dtype = torch.float32)
        self.T = torch.as_tensor(T, dtype = torch.float32)
        self.r = torch.as_tensor(r, dtype = torch.float32)
        self.sigma = torch.as_tensor(sigma, dtype = torch.float32)
        self.q = torch.as_tensor(q, dtype = torch.float32)

        # initialize number of steps in spot (M) and number of steps in time (N)
        
        self.M = M
        self.N = N

        self.S_max = 2 * self.K
        self.dt = self.T / self.N
        self.dS = self.S_max / self.M

        self.S = torch.linspace(0.0, self.S_max, self.M + 1, dtype=torch.float32)
        self.t = torch.linspace(0.0, self.T, self.N + 1, dtype=torch.float32)

        # creating member to store option prices to compare our PINN solution with the analytic solution
        
        self.C = torch.zeros((N + 1, M + 1))
        self.C[:, 0] = torch.as_tensor(0, dtype = torch.float32)
        self.C[-1,:] = torch.maximum(self.S - self.K, torch.tensor(0.0))

        t_grid, S_grid = torch.meshgrid(self.t[:-1], self.S[1:], indexing="ij")
        self.C[:-1, 1:] = self.priceOption(S_grid, t_grid)

        # creating member to store option deltas and gammas to filter data via greeks
    
        t_grid, S_grid = torch.meshgrid(self.t[:-1], self.S, indexing="ij")
        self.gamma = torch.zeros((N, M + 1))
        self.gamma = self.calcGammaOption(S_grid, t_grid)

        self.delta = torch.zeros_like(self.gamma)
        self.delta = self.calcDeltaOption(S_grid, t_grid)
        
    def priceOption(self, S, t):
        tau = torch.as_tensor(self.T - t, dtype=torch.float64)
        sqrt_tau = torch.sqrt(tau)
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        d2 = d1 - self.sigma * sqrt_tau
        norm = torch.distributions.Normal(0, 1)
        return S * torch.exp(-self.q * tau) * norm.cdf(d1) - K * torch.exp(-r * tau) * norm.cdf(d2)

    def calcGammaOption(self, S, t):
        tau = self.T - t
        sqrt_tau = torch.sqrt(tau)
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        norm = torch.distributions.Normal(0, 1)
        return torch.exp(-self.q * tau) * norm.log_prob(d1).exp() / (S * self.sigma * sqrt_tau)

    def calcDeltaOption(self, S, t):
        tau = self.T - t
        sqrt_tau = torch.sqrt(tau)
        d1 = (torch.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        norm = torch.distributions.Normal(0, 1)
        return torch.exp(-self.q * tau) * norm.cdf(d1)
    
    def filterDataByGamma(self, lowerGammaBound, upperGammaBound):
        mask = (lowerGammaBound <= self.gamma) & (self.gamma <= upperGammaBound)
        t_grid, S_grid = torch.meshgrid(self.t[:-1], self.S, indexing="ij")
        grid = torch.stack((t_grid, S_grid), dim=2)
        return grid[mask], self.C[ : -1, :][mask]

    def filterDataByDelta(self, lowerDeltaBound, upperDeltaBound):
        mask = (lowerDeltaBound <= self.delta) & (self.delta <= upperDeltaBound)
        t_grid, S_grid = torch.meshgrid(self.t[:-1], self.S, indexing="ij")
        grid = torch.stack((t_grid, S_grid), dim=2)
        return grid[mask], self.C[ : -1, :][mask]