import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_cifar10(batch_size=128, use_quick_test=False):
    """
    Load and preprocess CIFAR10 dataset
    
    Args:
        batch_size (int): Batch size for data loaders
        use_quick_test (bool): If True, use a small subset of data for quick testing
        
    Returns:
        dict: Dictionary containing train and validation data loaders
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_val, download=True)
    
    if use_quick_test:
        # Use only a small subset of data for quick test
        train_dataset.data = train_dataset.data[:1000]
        val_dataset.data = val_dataset.data[:500]
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    return dataloaders

def prepare_synthetic_quadratic():
    """
    Prepare synthetic quadratic optimization problem
    
    Returns:
        tuple: (A, b, x_init) where A is the quadratic matrix, b is the linear term, 
              and x_init is the initial parameter vector
    """
    torch.manual_seed(42)
    
    # Define quadratic loss: f(x) = 0.5 x^T A x + b^T x
    n = 10
    eigenvalues = torch.linspace(1, 10, n)
    Q, _ = torch.qr(torch.randn(n, n))
    A = Q @ torch.diag(eigenvalues) @ Q.t()
    b = torch.randn(n)
    
    # Initialize parameter vector
    x_init = torch.randn(n, requires_grad=True)
    
    return A, b, x_init

def load_wikitext2(batch_size=20, bptt=35, use_quick_test=False):
    """
    Load and preprocess WikiText-2 dataset for language modeling
    
    Args:
        batch_size (int): Batch size for data loaders
        bptt (int): Sequence length for language modeling
        use_quick_test (bool): If True, use a small subset of data for quick testing
        
    Returns:
        tuple: (train_data, valid_data, test_data, vocab) containing processed data and vocabulary
    """
    try:
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError:
        raise ImportError("Please install torchtext to use WikiText-2 dataset")
    
    tokenizer = get_tokenizer("basic_english")
    
    # Build vocabulary from the training split
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    train_iter = WikiText2(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    def data_process(raw_text_iter):
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        data = tuple(filter(lambda t: t.numel() > 0, data))
        return torch.cat(data)
    
    # Batchify data
    def batchify(data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        return data.view(bsz, -1).t().contiguous()
    
    # Reload iterators
    train_iter, valid_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    valid_data = data_process(valid_iter)
    test_data = data_process(test_iter)
    
    if use_quick_test:
        # Limit the data for quick test
        train_data = train_data[:1000]
        valid_data = valid_data[:500]
        test_data = test_data[:500]
    
    train_data = batchify(train_data, batch_size)
    valid_data = batchify(valid_data, batch_size)
    test_data = batchify(test_data, batch_size)
    
    return train_data, valid_data, test_data, vocab
