import torch
import torch.nn as nn
from typing import Optional, Tuple

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleRNN(nn.Module):
    """Simple RNN for PTB."""
    def __init__(self, vocab_size: int, embed_size: int = 300, hidden_size: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional tuple of (h_0, c_0) initial hidden states
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, sequence_length, vocab_size)
                - Tuple of final (h_n, c_n) hidden states
        """
        x = self.embedding(x)
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden
