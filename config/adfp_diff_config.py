"""
Configuration file for ADFP-Diff experiment.
"""

MODEL_CONFIG = {
    'hidden_channels': 64,
    'teacher_steps': 10,  # Fixed point iterations for teacher
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'image_size': 32,
    'num_workers': 4,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'teacher_epochs': 5,
    'student_epochs': 5,
    'aux_weight': 0.1,  # Weight for auxiliary loss
}

TEST_CONFIG = {
    'batch_size': 64,
    'constrained_steps': 1,  # Number of steps in resource-constrained mode
    'unconstrained_steps': 10,  # Number of steps in unconstrained mode
}

DATA_CONFIG = {
    'train_size': 1000,  # Size of dummy training dataset
    'val_size': 200,     # Size of dummy validation dataset
}

PATHS = {
    'models_dir': 'models',
    'teacher_model': 'models/teacher_model.pth',
    'student_model': 'models/student_model.pth',
    'student_aux_model': 'models/student_aux_model.pth',
    'logs_dir': 'logs',
    'plots': {
        'teacher_loss': 'logs/teacher_loss.pdf',
        'student_loss': 'logs/student_loss.pdf',
        'student_aux_loss': 'logs/student_aux_loss.pdf',
        'inference_time': 'logs/inference_time.pdf',
        'memory_usage': 'logs/memory_usage.pdf',
        'output_comparison': 'logs/output_comparison.pdf',
    }
}
