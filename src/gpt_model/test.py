from model.gpt import GPTLanguageModel
import torch
from config import ModelConfig

with open("../../data/Classroom of the Elite novels 1st year.txt" , "r" , encoding='utf-8') as f :
  
  text = f.read() 

chars = sorted(list(set(text))) 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

config = ModelConfig()

# Recreate the model with the correct vocab size
model2= GPTLanguageModel(config, vocab_size)
# Load weights, specifying device map
model2.load_state_dict(torch.load('../../model_weights/transformer_model_weights.pth', map_location=config.device))
# Move the entire model to the correct device
model2 = model2.to(config.device)

# Encode the starting text
# context = torch.tensor([[ModelConfig.vocab[c] for c in "Hi"]], dtype=torch.long, device=ModelConfig.device)

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
generated_text = model2.generate(context, max_new_tokens=1000)
print(decode(generated_text[0].tolist()))