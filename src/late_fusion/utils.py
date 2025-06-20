from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FusionDataset(Dataset):
     
     def __init__(self,
                  tab_logits,
                  met_logits,
                  labels):
          
          self.tab_logits = tab_logits
          self.met_logits = met_logits
          self.y = labels

    def __len__(self):
        return len(self.y)
    
    def __getitem__ (self,
                    idx):
        fused = torch.cat([self.tab_logits[idx], self.met_logits[idx]], dim=1)
        return fused, self.y[idx]


