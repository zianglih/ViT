import torch.nn as nn
import torch
import math

class PatchEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.projection = nn.Conv2d(in_channels=config["num_channels"],
                                    out_channels=config["hidden_size"],
                                    kernel_size=config["patch_size"],
                                    stride=config["patch_size"])

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2

        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, config["hidden_size"]))

    def forward(self, x):
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x


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
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return (x, attention_probs)
        

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config["num_hidden_layers"]
        self.use_patch_merger = config["use_patch_merger"]
        self.patch_merger_index = self.num_hidden_layers // 2

        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)
            
        self.patch_merger = PatchMerger(dim=config["hidden_size"], num_tokens_out=8)

    def forward(self, x):
        all_attentions = []
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
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)

class ViTForClassfication(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

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

