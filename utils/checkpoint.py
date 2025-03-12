import torch
import os

def save_checkpoint(model, path):
    """Save model checkpoint."""
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path):
    """Load model checkpoint."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Checkpoint loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")