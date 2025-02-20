import torch
import json
import numpy as np
from pathlib import Path
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator
from models import SimpleCNN, PTBLanguageModel
from train import get_cifar10_loaders, train_with_optimizer
import math
from optimizers import HybridOptimizer, AggMoOptimizer, MADGRADOptimizer

def get_ptb_data(batch_size=32):
    TEXT = Field(lower=True, batch_first=True)
    
    train, val, test = data.TabularDataset.splits(
        path='.',
        train='ptb.train.txt',
        validation='ptb.valid.txt',
        test='ptb.test.txt',
        format='text',
        fields=[('text', TEXT)]
    )
    
    TEXT.build_vocab(train)
    
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test),
        batch_size=batch_size,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False
    )
    
    return train_iter, val_iter, test_iter, TEXT.vocab

def train_with_optimizer_ptb(opt_name, model, train_data, val_data, test_data, device, epochs=10, lr=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    bptt = 35  # sequence length
    
    if opt_name == 'hybrid':
        optimizer = HybridOptimizer(model.parameters(), lr=lr)
    elif opt_name == 'aggmo':
        optimizer = AggMoOptimizer(model.parameters(), lr=lr)
    elif opt_name == 'madgrad':
        optimizer = MADGRADOptimizer(model.parameters(), lr=lr)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {opt_name}')
    
    train_losses, train_ppls = [], []
    val_losses, val_ppls = [], []
    
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target
    
    def evaluate(data_source):
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                data, targets = data.to(device), targets.to(device)
                output, _ = model(data)
                output_flat = output.view(-1, output.size(-1))
                total_loss += criterion(output_flat, targets).item() * targets.size(0)
        return total_loss / (data_source.size(0) - 1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch % 200 == 0:
                cur_loss = total_loss / 200
                print(f'| epoch {epoch:3d} | {batch:5d}/{train_data.size(0)//bptt:5d} batches '
                      f'| loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                total_loss = 0
        
        val_loss = evaluate(val_data)
        train_loss = evaluate(train_data)
        
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        train_losses.append(train_loss)
        train_ppls.append(train_ppl)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | train loss {train_loss:5.2f} | '
              f'train ppl {train_ppl:8.2f} | val loss {val_loss:5.2f} | '
              f'val ppl {val_ppl:8.2f}')
        print('-' * 89)
    
    test_loss = evaluate(test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}')
    print('=' * 89)
    
    return {
        'train_losses': train_losses,
        'train_ppls': train_ppls,
        'val_losses': val_losses,
        'val_ppls': val_ppls,
        'final_test_loss': test_loss,
        'final_test_ppl': test_ppl
    }

def evaluate_optimizers(task='cifar10', epochs=10, batch_size=128, lr=0.01, save_dir='results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    optimizers = ['hybrid', 'aggmo', 'madgrad', 'sgd', 'adam']
    
    if task == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size)
        model_class = SimpleCNN
        model_kwargs = {}
    else:  # ptb
        train_data, val_data, test_data, vocab = get_ptb_data(batch_size)
        model_class = PTBLanguageModel
        model_kwargs = {'vocab_size': len(vocab)}
    results = {}
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for opt_name in optimizers:
        print(f'\n{"="*50}\nEvaluating {opt_name.upper()} optimizer on {task}\n{"="*50}')
        model = model_class(**model_kwargs).to(device)
        
        if task == 'cifar10':
            opt_results = train_with_optimizer(
                opt_name, model, train_loader, test_loader, device,
                epochs=epochs, lr=lr
            )
        else:  # ptb
            opt_results = train_with_optimizer_ptb(
                opt_name, model, train_data, val_data, test_data,
                device, epochs=epochs, lr=lr
            )
        
        results[opt_name] = {
            'final_train_acc': opt_results['train_accs'][-1],
            'final_test_acc': opt_results['test_accs'][-1],
            'best_test_acc': max(opt_results['test_accs']),
            'convergence_epoch': np.argmax(opt_results['test_accs']) + 1,
            'train_loss_trend': opt_results['train_losses'],
            'test_loss_trend': opt_results['test_losses'],
            'train_acc_trend': opt_results['train_accs'],
            'test_acc_trend': opt_results['test_accs']
        }
        
        # Save model checkpoint
        torch.save(model.state_dict(), save_dir / f'{opt_name}_model.pth')
        
        # Print detailed results
        print(f'\nResults for {opt_name.upper()} optimizer:')
        print(f'Final Training Accuracy: {results[opt_name]["final_train_acc"]:.2f}%')
        print(f'Final Test Accuracy: {results[opt_name]["final_test_acc"]:.2f}%')
        print(f'Best Test Accuracy: {results[opt_name]["best_test_acc"]:.2f}%')
        print(f'Convergence Epoch: {results[opt_name]["convergence_epoch"]}')
        
    # Save all results to JSON
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Compare optimizers
    print('\nOptimizer Comparison:')
    print(f'{"Optimizer":<10} {"Final Test Acc":<15} {"Best Test Acc":<15} {"Convergence Epoch"}')
    print('-' * 55)
    for opt_name, opt_results in results.items():
        print(f'{opt_name:<10} {opt_results["final_test_acc"]:<15.2f} '
              f'{opt_results["best_test_acc"]:<15.2f} {opt_results["convergence_epoch"]}')
    
    return results

if __name__ == '__main__':
    evaluate_optimizers(epochs=2)  # Short test run
