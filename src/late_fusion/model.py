import torch
from torch import nn
from typing import List, Tuple, Dict, Any

from src.tab_transformer.tab_model import SAINT
from src.met_transformer.met_model import Stormer

class DeepSummit (nn.Module):
    
    def __init__(self,
                 tabular_model_path: str,
                 weather_model_path: str,
                 num_classes: int,
                 freeze_layers: bool = True,
                 device: torch.device = None
                 ):
        
            super.__init__()
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # Loading the model's saved weights from their trained .pth files
            self.saint   = torch.load(tabular_model_path, map_location=device).to(device)
            self.stormer = torch.load(weather_model_path,  map_location=device).to(device)

            if freeze_layers:
                  for parameter in self.saint.parameters(): parameter.requires_grad = False
                  for parameter in self.stormer.parameters(): parameter.requires_grad = False

            self.fusion = nn.Sequential(
                  nn.Linear(2 * num_classes, 2* num_classes),
                  nn.ReLU(),
                  nn.Dropout(0.2),
                  nn.Linear(2 * num_classes, num_classes)
            ).to(device)
           

    def forward(self,
                X_tab,
                X_weather):
          
          logits_tab = self.saint(X_tab)
          logits_weather = self.stormer(X_weather)
          cat = torch.cat([logits_tab, logits_weather], dim=1)
          fused = self.fusion(cat)
          return fused