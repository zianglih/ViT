# Vision Transformer for Image Recognition

## Members

Hanlin Bi (hanlinbi)
Junwen Yu (junwenyu)
Ziang Li (ziangli)
Zihao Ye (zihaoye)

## Group Dynamics

We plan to discuss through a mixture of online chats and on-site meetings. We plan to divide the task into several stages, and meet for 2-4 times each stage, to keep everyone in pace.

## Problem Statement

Transformer architecture has showed its great potential in Natural Language Processing tasks. In this project, we are going to explore its application in computer vision tasks, specifically, image recognition, by implementing a simplified version of the vision transformer (ViT) architecture. We will evaluate its performance and compare it with traditional CNN approaches, and attempt to discuss why or why not it's a good fit for such tasks. Moreover, we are also interested in studying further research regarding ViT, like the Masked Autoencoder (MAE). We will try to modify our model based on such research, and evaluate the performance.

## Approach

We plan to working on 2 tasks: the implementation and evaluation of the original ViT, and then the followup, e.g., MAE. For each task we have 2 students. For the ViT part, we will directly train and tune the model on small, labeled datasets, then try to generalize the model by training on larger datasets. For the MAE part, we are likely to work on unlabeled, relatively large datasets, fine-tune for the small dataset, and compare with the original approach.

## Datasets

Small: CIFAR-10 (60000 3x32x32 10 classes)
Relatively large: Tiny ImageNet (100000 3x64x64 200 classes), CIFAR-100, etc.

## Computational Resources

Colab, 3060, 4050. We may rent H100 if needed.

## Evaluation

- ViT vs CNN
- MAE transfer learning vs ViT direct training
