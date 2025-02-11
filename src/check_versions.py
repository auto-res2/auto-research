import torch
import torchvision
import torchtext
from utils.optimizers import HybridOptimizer

def main():
    print('Package versions:')
    print(f'torch: {torch.__version__}')
    print(f'torchvision: {torchvision.__version__}')
    print(f'torchtext: {torchtext.__version__}')
    print('\nAll required packages are installed successfully.')

if __name__ == '__main__':
    main()
