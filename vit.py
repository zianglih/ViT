import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        patch_size = config["patch_size"]
        num_channels = config["num_channels"] 

        self.to_patches = nn.Conv2d(in_channels=num_channels,
                                    out_channels=config["hidden_size"],
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, x):
        # B C H W
        x = self.to_patches(x)
        # B hidden_size image_size/patch_size image_size/patch_size
        x = x.flatten(2)
        # B hidden_size (image_size/patch_size)**2 = num_patches
        x = x.transpose(1, 2)
        # B num_patches hidden_size
        return x

class ShiftedPatchEmbeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.patch_dim = self.patch_size * self.patch_size * self.num_channels * 5

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.hidden_size)
        )

    def forward(self, x):
        shifts = [(1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1)]
        shifted_x = [F.pad(x, shift, mode='circular') for shift in shifts]
        x_with_shifts = torch.cat([x] + shifted_x, dim=1)
        return self.to_patch_tokens(x_with_shifts)

def gen_pos_embedding(num_patches, hidden_size):
    position_enc = torch.zeros(num_patches, hidden_size)
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

    position_enc[:, 0::2] = torch.sin(position * div_term)
    position_enc[:, 1::2] = torch.cos(position * div_term)

    position_enc = position_enc.unsqueeze(0)
    return nn.Parameter(position_enc, requires_grad=False)

class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        hidden_size = config["hidden_size"] 
        self.config = config
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2

        self.patch_embeddings = PatchEmbeddings(config)
        # cls: 1 1 hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        if (config["use_simpleViT"] == 0):
            self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        elif (config["use_simpleViT"] == 1):
            self.position_embeddings = gen_pos_embedding(self.num_patches, hidden_size)
        # self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        # cls: B 1 hidden_size
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if (self.config["use_simpleViT"] == 0):
            x = x + self.position_embeddings
        elif (self.config["use_simpleViT"] == 1):
            x[:, 1:] += self.position_embeddings
        # x = self.dropout(x)
        return x

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.hidden_size),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=config["hidden_size"],
        #     num_heads=config["num_attention_heads"],
        #     dropout=config["attention_probs_dropout_prob"],
        #     batch_first=True
        # )
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        # attention_output, attention_probs = self.attention(query = x, key = x, value = x)
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return (x, attention_probs)
  
# from performer_pytorch import Performer

# class EfficientSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.performer = Performer(
#             dim=config["hidden_size"],
#             depth=1,
#             heads=config["num_attention_heads"],
#             causal=True,
#             dim_head=config["hidden_size"] // config["num_attention_heads"],
#         )

#     def forward(self, x):
#         return self.performer(x)

class AttentionHead(nn.Module):

    def __init__(self, hidden_size, attention_head_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
    

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.qkv_bias = config["qkv_bias"]

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                self.qkv_bias
            )
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
        return (attention_output, attention_probs)

class LocalSelfAttension(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["hidden_size"]
        heads = config["num_attention_heads"]
        dim_head = dim // heads
        dropout = config["attention_probs_dropout_prob"]
        self.use_performer = config["use_performer"]

        inner_dim = dim_head *  heads
        # self.efficent_self_attention = EfficientSelfAttention(config)
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        # if self.use_performer:
        #     x = self.efficent_self_attention(x)
        q, k, v = self.prepare_qkv(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = float('-inf')
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def prepare_qkv(self, x):
        qkv = self.to_qkv(x)
        return rearrange(qkv, 'b n (h d three) -> three b h n d', h=self.heads, three=3)

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config["num_hidden_layers"]
        self.use_patch_merger = config["use_patch_merger"]
        self.patch_merger_index = self.num_hidden_layers // 2
        self.use_local_self_attention = config["use_local_self_attention"]
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            if self.use_local_self_attention:
                lsa = LocalSelfAttension(config)
                self.blocks.append(nn.ModuleList([block, lsa]))
            else:
                self.blocks.append(block)
        self.patch_merger = PatchMerger(dim=config["hidden_size"], num_tokens_out=8)

    def forward(self, x):
        all_attentions = []
        if self.use_local_self_attention:
            for block, lsa in self.blocks:
                x, attention_probs = block(x)
                x = lsa(x) + x
                all_attentions.append(attention_probs)
            return (x, all_attentions)

        for i, block in enumerate(self.blocks):
            x, attention_probs = block(x)
            all_attentions.append(attention_probs)
                
            if self.use_patch_merger and i == self.patch_merger_index-1:
                x = self.patch_merger(x)
        return (x, all_attentions)
        
class FinalLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.last_layer_use_mlp = config["last_layer_use_mlp"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        if (self.last_layer_use_mlp):
            self.classifier = MLP(config)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_classes)
            
    def forward(self, x):
        return self.classifier(x)
        

class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1) * (dim ** -0.5))  # Learnable scale
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))
        self.bias = nn.Parameter(torch.zeros(num_tokens_out, 1))  # Bias
        self.dropout = nn.Dropout(0.1)
        self.temp = nn.Parameter(torch.ones(1))  # Learnable T

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale + self.bias
        attn = (sim / self.temp).softmax(dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, x)

class ViTForClassfication(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.use_shifted_patch_embeddings = config["use_shifted_patch_embeddings"]
        # Create the embedding module
        if self.use_shifted_patch_embeddings:
            self.embedding = ShiftedPatchEmbeddings(config)
        else:
            self.embedding = Embeddings(config)

        self.encoder = Encoder(config)
        self.classifier = FinalLayer(config)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=config["initializer_range"])
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                continue
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                continue
            if isinstance(module, Embeddings):
                module.position_embeddings.data = nn.init.trunc_normal_(
                    module.position_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=config["initializer_range"],
                ).to(module.position_embeddings.dtype)
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data.to(torch.float32),
                    mean=0.0,
                    std=config["initializer_range"],
                ).to(module.cls_token.dtype)
                continue

    def forward(self, x):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output)
        logits = self.classifier(encoder_output[:, 0])
        return (logits, all_attentions)

