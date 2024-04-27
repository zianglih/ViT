import torch.nn as nn
import torch
import math

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    """
    Add positional embeddings to the patch embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create patch embeddings
        self.patch_embeddings = PatchEmbeddings(config)
        self.hidden_size = config["hidden_size"]
        # Create a learnable parameter.
        # It is designed to accumulate representational information for classification tasks. 
        # It's initialized with random values and is optimized during training.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        # Create position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.hidden_size))

    def forward(self, x):
        # Convet image x to patches in hidden dimensions
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]
        # Match the size of the cls token to the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the cls token to the input
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embeddings
        x = x + self.position_embeddings
        # Now we have the input ready for the transformer
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    """
    def __init__(self, hidden_size, attention_head_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
    
    def forward(self, x):
        # Get the query, key, and value projections for all tokens
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Get attension score
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # Calculate attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)
    

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

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
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        

class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)
            
        # Initialize the Patch Merger at middle layer
        self.num_hidden_layers = config["num_hidden_layers"]
        self.use_patch_merger = config["use_patch_merger"]
        self.patch_merger_index = self.num_hidden_layers // 2
        self.patch_merger = PatchMerger(dim=config["hidden_size"], num_tokens_out=8)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for i, block in enumerate(self.blocks):
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
                
            # Apply patch merger:
            if self.use_patch_merger and i == self.patch_merger_index-1:
                x = self.patch_merger(x)

        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
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
        

# patch merger class
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
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = FinalLayer(config)

        # Initialize the weights
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
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
                    std=self.config["initializer_range"],
                ).to(module.position_embeddings.dtype)
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data.to(torch.float32),
                    mean=0.0,
                    std=self.config["initializer_range"],
                ).to(module.cls_token.dtype)
                continue

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

