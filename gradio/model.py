from embed import embed
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim = 100, hidden_dim = 32, num_layers =8, output_dim = 3, batch_size=32):
        super(LSTMModel, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        


    def forward(self, text, pro = False, predict = False, batched = True):
        

        if pro:
          text = embed(text)
        


        if batched:
          (h0, c0) = self.init_hidden(1)  
          
        else:
          h0, c0 = self.init_hidden(1)  
          h0 = torch.zeros(self.num_layers, self.hidden_dim)
          c0 = torch.zeros(self.num_layers, self.hidden_dim)
          

        output, (hn, cn) = self.lstm(text, (h0, c0))

        final_hidden = hn[-1, :]
        out = self.fc(final_hidden)
        out = F.softmax(out, -1)


        if predict == False:
          return out
        else:
          return torch.argmax(out) - 1

    def init_hidden(self, batch_size):
      h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
      c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

      return (h0, c0)
