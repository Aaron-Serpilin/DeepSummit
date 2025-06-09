import torch
from torch import nn
from typing import List, Tuple, Dict

class DeepSummit (nn.Module):
    
    def __init__(self,
                 tabular_model_path: str,
                 weather_model_path: str,
                 num_classes: int,
                 freeze_layers: bool = True,
                 kwargs: Dict[str, Any] = None
                 ):
        
            super.__init__()
            # self.saint = SAINT(...)
            self.saint = 0
            # self.stormer = Stormer(...)
            self.stormer = 0

            self.kwargs = kwargs or {}

            self.saint.load_state_dict(torch.load(tabular_model_path))
            self.stormer.load_state_dict(torch.load(weather_model_path))

            if freeze_layers:
                  for parameter in self.saint.parameters(): parameter.requires_grad = False
                  for parameter in self.stormer.parameter(): parameter.requires_grad = False

            self.fusion = nn.Linear(2 * num_classes, num_classes)

    def forward(self,
                X_tab,
                X_weather):
          
          logits_tab = self.saint(X_tab)
          logits_weather = self.stormer(X_weather)
          cat = torch.cat([logits_tab, logits_weather], dim=1)
          fused = self.fusion(cat)
          return fused