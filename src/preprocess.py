import torch
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext

def get_cifar10_transforms():
    """Get the transforms for CIFAR-10 dataset."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return train_transform, test_transform

def get_ptb_processor(batch_size=32):
    """Get the data processor for PTB dataset."""
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    def build_vocab():
        train_iter = torchtext.datasets.PennTreebank(split='train')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    
    def process_raw_data(raw_text_iter, vocab):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
                for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    
    def batchify(data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size]
        data = data.view(batch_size, -1).t().contiguous()
        return data
    
    return {
        'tokenizer': tokenizer,
        'build_vocab': build_vocab,
        'process_raw_data': process_raw_data,
        'batchify': batchify
    }

if __name__ == '__main__':
    # Test CIFAR-10 transforms
    train_transform, test_transform = get_cifar10_transforms()
    print("CIFAR-10 transforms created successfully")
    
    # Test PTB processor
    ptb_processor = get_ptb_processor()
    vocab = ptb_processor['build_vocab']()
    print(f"PTB vocabulary size: {len(vocab)}")
