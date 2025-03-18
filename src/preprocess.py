import torch
import torchvision
import torchvision.transforms as transforms
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

def get_cifar10_data(batch_size=128, num_workers=2):
    """
    Load and preprocess CIFAR-10 dataset
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data transforms for CIFAR-10
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
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_agnews_data(batch_size=64):
    """
    Load and preprocess AG_NEWS dataset
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader, vocab)
    """
    # Load AG_NEWS dataset
    train_iter, test_iter = AG_NEWS(split=('train', 'test'))
    
    # Convert iterators to lists for multiple passes
    train_list = list(train_iter)
    test_list = list(test_iter)
    
    # Build vocabulary
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
            
    vocab = build_vocab_from_iterator(
        yield_tokens(train_list), 
        specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    
    # Define collate function for data loader
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(_label - 1)  # AG_NEWS labels are 1-indexed
            processed_text = torch.tensor(
                vocab(tokenizer(_text)), 
                dtype=torch.int64
            )
            text_list.append(processed_text)
        
        # Pad sequences to same length
        text_list = torch.nn.utils.rnn.pad_sequence(
            text_list, 
            batch_first=True
        )
        
        return torch.tensor(label_list), text_list
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_list, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    test_loader = torch.utils.data.DataLoader(
        test_list, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    return train_loader, test_loader, vocab

def create_rosenbrock_function():
    """
    Create the Rosenbrock function for optimization testing
    
    Returns:
        function: Rosenbrock function that takes a tensor of shape (2,)
    """
    def rosenbrock(x):
        """
        Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        where a=1, b=100 are constants
        
        Args:
            x (torch.Tensor): Tensor of shape (2,) containing [x, y]
            
        Returns:
            torch.Tensor: Scalar tensor containing the function value
        """
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    return rosenbrock
