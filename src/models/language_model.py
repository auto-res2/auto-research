import torch
import torch.nn as nn

class PTBLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=200, num_layers=2, dropout=0.5):
        super(PTBLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.fc(output)
        return decoded, hidden
