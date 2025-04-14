"""
Model definitions for the Progressive Brightness Distillation Diffusion experiment.
"""

import torch
import torch.nn as nn
from config.pbd_diffusion_config import TEACHER_CHANNELS, STUDENT_CHANNELS

class InitialBrightnessCorrection(nn.Module):
    def __init__(self):
        super(InitialBrightnessCorrection, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
    
    def forward(self, x):
        return x + self.bias

class ProgressiveRefinement(nn.Module):
    def __init__(self):
        super(ProgressiveRefinement, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x, teacher_correction):
        student_update = self.conv(x)
        return x + student_update + teacher_correction

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, STUDENT_CHANNELS, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(STUDENT_CHANNELS, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)
