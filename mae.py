import torch
from torch import nn, optim
from vit import ViTForClassfication

from utils import save_experiment, save_checkpoint
from data import prepare_data

from vit import Encoder
from train import Trainer

import json

metadata = {}
with open('parameters.json') as f:
    metadata = json.load(f)

config = metadata['config']
args = metadata['args']
device = metadata['args']['device']

def gen_config(patch_size = 4,
        hidden_size = 48,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        intermediate_size = 192,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
        initializer_range = 0.02,
        image_size = 32,
        num_classes = 10, 
        num_channels = 3,
        qkv_bias = 1,
        use_faster_attention = 1):

    config = {"patch_size": patch_size,
              "hidden_size": hidden_size,
              "num_hidden_layers": num_hidden_layers,
              "num_attention_heads": num_attention_heads,
              "intermediate_size": intermediate_size,
              "hidden_dropout_prob": hidden_dropout_prob,
              "attention_probs_dropout_prob": attention_probs_dropout_prob,
              "initializer_range": initializer_range,
              "image_size": image_size,
              "num_classes": num_classes,
              "num_channels": num_channels,
              "qkv_bias": qkv_bias,
              "use_faster_attention": use_faster_attention}

    return config

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, 
                 mask_ratio = 0.75, decoder_depth = 1):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.config = gen_config(
            num_hidden_layers=decoder_depth, 
            hidden_size=decoder_dim,
            intermediate_size=decoder_dim * 4)
        num_patches_plus_one, encoder_dim = encoder.embedding.position_embeddings.shape[-2:]
        num_patches = num_patches_plus_one - 1
        # align dimensions between the encoder and decoder
        self.align_enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Encoder(self.config)

        self.decoder_position_embeddings = nn.Embedding(num_patches, decoder_dim)
        self.projection = nn.Linear(decoder_dim, encoder.embedding.patch_embeddings.patch_size ** 2 \
                                    * encoder.embedding.patch_embeddings.hidden_size)
        
    def forward(self, x):
        patches = self.encoder.embedding.patch_embeddings(x)
        batch, num_patches = patches.shape[:2]
        # tokens = self.encoder.embedding.


def main():
    model = ViTForClassfication(config)
    mae = MAE(model, 512)

    batch_size = args["batch_size"]
    epochs = args["epochs"]
    lr = args["lr"]
    device = args["device"]
    save_model_every_n_epochs = args["save_model_every"]
    # Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    main()