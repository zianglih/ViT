# Vision Transformer for Image Recognition

## Members

Hanlin Bi (hanlinbi)

Junwen Yu (junwenyu)

Ziang Li (ziangli)

Zihao Ye (zihaoye)

## Group Dynamics

We plan to discuss through a mixture of online chats and on-site meetings. We plan to divide the task into several stages, and meet for 2-4 times each stage, to keep everyone in pace. Task assignments will vary depending on individual's strengths and interests.

## Problem Statement

Transformer architecture has showed its great potential in Natural Language Processing tasks. In this project, we are going to explore its application in computer vision tasks, specifically, image recognition, by implementing a simplified version of the vision transformer (ViT) architecture. We foresee the requirements for computational resources during training a large transformer model so we decide to implement one with smaller paramenter size. We will evaluate its performance and compare it with traditional CNN approaches, and attempt to discuss why or why not it's a good fit for such tasks. Moreover, we are also interested in studying Masked Autoencoder (MAE), which is a further research based on ViT. 

## Approach

We plan to working on 2 tasks: the implementation and evaluation of the original ViT, and then the MAE followup.

For the ViT part, we will implement it from scratch, mainly using PyTorch. Then we will train the model acroos datasets of different scales, verify the results and compare the results with CNN.

For the MAE followup, we will use checkpoints from the pretrained models, and fine-tune it on our model. Note that, if we feel this part is not that relavant, we may only focus on the vit part, e.g., do more experiments.

## Datasets

- Small: 
  - CIFAR-10 (60000 3x32x32 10 classes)

- Relatively large: 
  - Tiny ImageNet (100000 3x64x64 200 classes)
  - CIFAR-100, etc.

## Computational Resources

Colab, 3060, 4050. We may rent H100 if needed.

Ideally, the model should be able to converge on less than 2 hours of training on a RTX 4050 laptop GPU.

## Evaluation

- ViT vs. CNN, accuracy on different data scale
- Algorithm enhancement proposed by MAE and other following work
- MAE transfer learning vs. ViT direct training, compare based on speed/accuracy/supervised/...
