import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    Used by: Embeddings
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Embeddings(nn.Module):
    """
    Add positional embeddings to the patch embeddings.
    Use: PatchEmbeddings
    Used by: ViTForClassfication
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class AttentionHead(nn.Module):
    """
    A single attention head.
    Used by: MultiHeadAttention
    """

    def __init__(self, hidden_size, attention_head_size, bias=True):
        super().__init__()
        pass

    def forward(self, x):
        pass


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    Use: AttentionHead
    Used by: Block
    """

    def __init__(self, config):
        pass

    def forward(self, x):
        pass


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    Used by: Block, FinalLayer
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Block(nn.Module):
    """
    A single transformer block.
    Use: MultiHeadAttention, MLP
    Used by: Encoder
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Encoder(nn.Module):
    """
    The transformer encoder module.
    Use: Block
    Used by: ViTForClassfication
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class FinalLayer(nn.Module):
    """
    The final layer.
    Used: MLP
    Used by: ViTForClassfication
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    Use: Embeddings, Encoder, FinalLayer
    """

    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, x):
        pass
