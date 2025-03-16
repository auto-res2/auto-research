import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader) for CIFAR-10
    """
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_ptb_loaders(batch_size=20, seq_length=35):
    """
    Load and preprocess Penn Treebank dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        seq_length (int): Sequence length for language modeling
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader, vocab_size) for PTB
    """
    # Load PTB dataset
    dataset = load_dataset("ptb_text_only")
    
    # Create vocabulary
    word_list = " ".join(dataset["train"]["sentence"]).split()
    vocab = sorted(set(word_list))
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    def tokenize_data(data):
        """Convert text to token indices"""
        tokenized = []
        for sentence in data["sentence"]:
            tokens = [word_to_idx[word] for word in sentence.split()]
            tokenized.extend(tokens)
        return tokenized
    
    # Tokenize datasets
    train_data = tokenize_data(dataset["train"])
    valid_data = tokenize_data(dataset["validation"])
    test_data = tokenize_data(dataset["test"])
    
    # Create batches
    def batchify(data, batch_size):
        """Divide the dataset into batch_size parts"""
        nbatch = data.size(0) // batch_size
        data = data.narrow(0, 0, nbatch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data
    
    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    valid_tensor = torch.tensor(valid_data, dtype=torch.long)
    test_tensor = torch.tensor(test_data, dtype=torch.long)
    
    # Batchify data
    train_batches = batchify(train_tensor, batch_size)
    valid_batches = batchify(valid_tensor, batch_size)
    test_batches = batchify(test_tensor, batch_size)
    
    # Create data loaders
    def get_batch(source, i, seq_length):
        """Get a batch for language modeling with context and target"""
        seq_len = min(seq_length, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    
    class LanguageModelDataLoader:
        def __init__(self, data, batch_size, seq_length):
            self.data = data
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.num_batches = (len(data) - 1) // seq_length
            
        def __iter__(self):
            for i in range(0, self.data.size(0) - 1, self.seq_length):
                yield get_batch(self.data, i, self.seq_length)
                
        def __len__(self):
            return self.num_batches
    
    train_loader = LanguageModelDataLoader(train_batches, batch_size, seq_length)
    valid_loader = LanguageModelDataLoader(valid_batches, batch_size, seq_length)
    test_loader = LanguageModelDataLoader(test_batches, batch_size, seq_length)
    
    return train_loader, valid_loader, test_loader, vocab_size
