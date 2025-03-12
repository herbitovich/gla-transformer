import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from models.transformer import GLATransformer
from data.dataset import get_dataloader
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, val_loader, criterion, logger):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=-1)
            total_correct += (predicted.view(-1) == targets.view(-1)).sum().item()
            total_samples += targets.numel()

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    logger.info(f"Validation | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

def main():
    config = {
        "vocab_size": 50257,
        "dim": 512,
        "num_layers": 4,
        "num_heads": 4,
        "batch_size": 8,
        "epochs": 1,
        "lr": 1e-4,
        "seq_len": 512,  # Must be divisible by chunk_size (64)
        "chunk_size": 64
    }
    
    logger = setup_logger()
    model = GLATransformer(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"]
    )

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = CrossEntropyLoss()
    train_loader = get_dataloader(config["batch_size"], split="train")  
    val_loader = get_dataloader(config["batch_size"], split="validation")
    
    logger.info("Starting training...")
    logger.info(f'The number of trainable parameters: {count_parameters(model)}')
    logger.info(f"The number of batches in the training dataset: {len(train_loader)}")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 1 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        evaluate(model, val_loader, criterion, logger)

        save_checkpoint(model, f"checkpoint_epoch{epoch+1}.pt")
        
if __name__ == "__main__":
    main()
