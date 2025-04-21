import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from src.tab_transformer.tab_attention import Transformer, RowColTransformer  
from src.tab_transformer.tab_blocks import MLP, simple_MLP, sep_MLP 

### SAINT Model ###

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories, # tuple of (#classes per categorical column)
        num_continuous, # number continuous features
        dim, # embedding dimension for each token
        depth, # number of transformer layers
        heads,# number of attention heads
        dim_head = 16, # per-head dimension
        dim_out = 1, # final output dimension
        mlp_hidden_mults = (4, 2), # multipliers for the final MLP hidden layers
        mlp_act = None, # activation final MLP
        num_special_tokens = 0,
        attn_dropout = 0., # attention dropout
        ff_dropout = 0., # feed-forward dropout
        cont_embeddings = 'MLP', # how to embed continuous
        attentiontype = 'col', 
        final_mlp_style = 'common',
        y_dim = 2 # output classes for classification
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'Number of each category must be positive'
      
        # --- 1. Category Metadata ---

        # Categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # Create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # print(f"[INIT] #categories={self.num_categories}, #unique_categories={self.num_unique_categories}, #special_tokens={self.num_special_tokens}")
        # print(f"[INIT] total_tokens (vocab size) = {self.total_tokens}")

        # For automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)

        # --- 2. Continuous Feature Setup ---

        self.num_continuous = num_continuous
        self.norm = nn.LayerNorm(num_continuous)
        self.cont_embeddings = cont_embeddings

        # print(f"[INIT] #continuous features = {self.num_continuous}, cont_embeddings = '{self.cont_embeddings}'")

        # --- 3. Intermediate Sizes ---

        self.dim = dim
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        # print(f"[INIT] embedding dim = {self.dim}, attentiontype = '{self.attentiontype}', final_mlp_style = '{self.final_mlp_style}'")

        if self.cont_embeddings == 'MLP':

            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous

        elif self.cont_embeddings == 'pos_singleMLP':

            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous

        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # print(f"[INIT] computed input_size for final MLP = {input_size}, nfeats = {nfeats}")

        # --- 4. Transformer Instantiation ---

        if attentiontype == 'col':

            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )

        elif attentiontype in ['row','colrow'] :

            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        # print(f"[INIT] Transformer built: {self.transformer.__class__.__name__}")

        # --- 5. Token & Mask Embeddings ---

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)

        # print(f"[INIT] Head MLP dims = {all_dimensions}")

        # Embeddings for categorical tokens
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        # Offsets for mask embeddings
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        # Mask embeddings
        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)

        # print(f"[INIT] Mask & position embeddings created")
        
        # --- 6. Final MLP Heads ---

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

    def forward(self, x_categ, x_cont):

        # print(f"[FORWARD] x_categ.shape = {x_categ.shape}, x_cont.shape = {x_cont.shape}")
        x = self.transformer(x_categ, x_cont)
        # print(f"[FORWARD] After transformer, x.shape = {x.shape}")

        cat_part = x[:, :, self.num_categories, :]
        cont_part = x[:, self.num_categoties :, :]
        # print(f"[FORWARD] cat_part.shape = {cat_part.shape}, cont_part.shape = {cont_part.shape}")

        cat_outs = self.mlp1(cat_part)
        con_outs = self.mlp2(cont_part)
        # print(f"[FORWARD] out_cat.shape = {cat_outs.shape}, out_con.shape = {con_outs.shape}")
        return cat_outs, con_outs 