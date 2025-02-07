from config import ModelConfig
from data_loader import TextDataset
from model.gpt import GPTLanguageModel
from trainer import Trainer
import torch

def main():
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize dataset
    dataset = TextDataset(config)
    dataset.load_data('../../data/Classroom of the Elite novels 1st year.txt')
    
    # Initialize model
    model = GPTLanguageModel(config, dataset.vocab_size)
    model = model.to(config.device)
    
    # Print model size
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Initialize trainer and train
    trainer = Trainer(model, dataset, config)
    trainer.train()
    
    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(dataset.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == "__main__":
    main()