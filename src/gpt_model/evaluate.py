import torch
import torch.nn.functional as F
from model.gpt import GPTLanguageModel
from config import ModelConfig

def get_batch(data, block_size, batch_size, device):
    """
    Generate a batch of data for evaluation
    
    Args:
        data (torch.Tensor): Full dataset
        block_size (int): Context length
        batch_size (int): Number of sequences in a batch
        device (torch.device): Computing device
    
    Returns:
        tuple: Input sequences and target sequences
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def evaluate_model(model, train_data, val_data, config):
    """
    Comprehensive model evaluation
    
    Args:
        model (GPTLanguageModel): Trained model
        train_data (torch.Tensor): Training dataset
        val_data (torch.Tensor): Validation dataset
        config (ModelConfig): Model configuration
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()  # Set model to evaluation mode
    device = config.device
    
    # Prepare evaluation metrics
    eval_results = {
        'train_loss': 0.0,
        'val_loss': 0.0,
        'train_perplexity': 0.0,
        'val_perplexity': 0.0
    }
    
    # Disable gradient computation
    with torch.no_grad():
        # Evaluate on training data
        train_losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(train_data, config.block_size, config.batch_size, device)
            _, loss = model(x, y)
            train_losses.append(loss.item())
        
        # Compute training metrics
        eval_results['train_loss'] = torch.tensor(train_losses).mean().item()
        eval_results['train_perplexity'] = torch.exp(torch.tensor(eval_results['train_loss'])).item()
        
        # Evaluate on validation data
        val_losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(val_data, config.block_size, config.batch_size, device)
            _, loss = model(x, y)
            val_losses.append(loss.item())
        
        # Compute validation metrics
        eval_results['val_loss'] = torch.tensor(val_losses).mean().item()
        eval_results['val_perplexity'] = torch.exp(torch.tensor(eval_results['val_loss'])).item()
    
    return eval_results

def main():
    # Load configuration and dataset
    config = ModelConfig()
    
    # Load text data
    with open("../../data/Classroom of the Elite novels 1st year.txt", "r", encoding='utf-8') as f:
        text = f.read()
    
    # Prepare dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode entire text
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Load model
    model = GPTLanguageModel(config, vocab_size)
    model.load_state_dict(torch.load('../../model_weights/transformer_model_weights.pth', map_location=config.device))
    model = model.to(config.device)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, train_data, val_data, config)
    
    # Print results
    print("Model Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()