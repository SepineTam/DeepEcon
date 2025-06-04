import torch


def generate_toy_data(n=200, p=5, seed=0):
    """Generate simple synthetic data for testing."""
    torch.manual_seed(seed)
    X = torch.randn(n, p)
    beta = torch.arange(1.0, p + 1.0)
    logits = (X @ beta * 0.1)
    T = torch.bernoulli(torch.sigmoid(logits))
    Y = X @ beta + 0.5 * T + torch.randn(n)
    return X, T, Y
