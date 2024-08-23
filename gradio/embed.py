import torch
from tokenizer import tokenizer
from torchtext.vocab import Vectors

max_len = 64

glove_file = 'glove.6B.100d.txt'

# Load the pre-trained GloVe embeddings from the local file
glove = Vectors(glove_file)

def embed(sentence):
  
  tok = tokenizer(sentence.lower()) # tokenization


  if len(tok) >= max_len:  # truncation
    tok = tok[1:max_len + 1]
  
  pad = max_len - len(tok)

  output = []
    
  for i in range(len(tok)):
    if not(tok[i].text in glove.stoi):
        pad = pad+1

    else:
      word_embedding = glove.vectors[glove.stoi[tok[i].text]]
      output.append(word_embedding)
  
  for i in range(pad): # padding
      output.append(torch.zeros(100))
  
  return torch.stack(output)

