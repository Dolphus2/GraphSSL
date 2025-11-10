import torch, random, numpy as np

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS device")
        return torch.device("mps")
    print("Using CPU device")
    return torch.device("cpu")

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0: return float("nan")
    pred = logits[idx].argmax(-1)
    return (pred == y[idx]).float().mean().item()

if __name__ == "__main__":
    set_seed(42, deterministic=True)
    device = resolve_device()
    print(f"Using device: {device}")