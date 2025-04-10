"""
Model definitions for the DITTO-GSD experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DITTOModel(nn.Module):
    """
    Baseline DITTO model with implicit fusion decoder.
    """
    def __init__(self, point_features=128, grid_features=64):
        super(DITTOModel, self).__init__()
        self.point_features = point_features
        self.grid_features = grid_features
        
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, point_features),
            nn.ReLU()
        )
        
        self.grid_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, grid_features),
            nn.ReLU()
        )
        
        self.fusion_decoder = nn.Sequential(
            nn.Linear(point_features + grid_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        point_features = self.point_encoder(x)
        
        grid_features = self.grid_encoder(x)
        
        fusion_features = torch.cat([point_features, grid_features], dim=-1)
        
        output = self.fusion_decoder(fusion_features)
        
        return output

class DITTOGSDModel(nn.Module):
    """
    DITTO-GSD model with Gaussian splatting decoder.
    """
    def __init__(self, point_features=128, grid_features=64, use_proj=True, use_gs_decoder=True):
        super(DITTOGSDModel, self).__init__()
        self.point_features = point_features
        self.grid_features = grid_features
        self.use_proj = use_proj
        self.use_gs_decoder = use_gs_decoder
        
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, point_features),
            nn.ReLU()
        )
        
        self.grid_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, grid_features),
            nn.ReLU()
        )
        
        if use_proj:
            self.projection = nn.Sequential(
                nn.Linear(point_features + grid_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        
        if use_gs_decoder:
            self.gs_params = nn.Sequential(
                nn.Linear(128 if use_proj else point_features + grid_features, 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # 3 for position, 3 for scale, 3 for rotation, 1 for opacity
            )
            
            self.gs_decoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )
        else:
            self.standard_decoder = nn.Sequential(
                nn.Linear(128 if use_proj else point_features + grid_features, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, x):
        point_features = self.point_encoder(x)
        
        grid_features = self.grid_encoder(x)
        
        combined_features = torch.cat([point_features, grid_features], dim=-1)
        
        if self.use_proj:
            features = self.projection(combined_features)
        else:
            features = combined_features
        
        if self.use_gs_decoder:
            gs_params = self.gs_params(features)
            
            position = gs_params[:, :3]
            scale = F.softplus(gs_params[:, 3:6])  # Ensure positive scale
            rotation = gs_params[:, 6:9]
            opacity = torch.sigmoid(gs_params[:, 9:10])  # Ensure opacity in [0, 1]
            
            output = self.gs_decoder(gs_params)
        else:
            output = self.standard_decoder(features)
        
        return output
