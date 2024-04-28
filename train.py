import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import os
import time

from data import prepare_data
from vit import ViTForClassfication

import torch
from torch import nn, optim

import json


class Trainer:
    """
    The transformer trainer
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.config = config

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        train_losses, test_losses, accuracies = [], [], []
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {
                  test_loss:.4f}, Accuracy: {accuracy:.4f}")
        outdir = os.path.join("./experiments", self.exp_name)
        os.makedirs(outdir, exist_ok=True)
        configfile = os.path.join(outdir, 'config.json')
        with open(configfile, 'w') as f:
            json.dump(self.config, f, sort_keys=True, indent=4)
        jsonfile = os.path.join(outdir, 'metrics.json')
        with open(jsonfile, 'w') as f:
            data = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies,
            }
            json.dump(data, f, sort_keys=True, indent=4)
        cpfile = os.path.join(outdir, f'model_{epochs}.pt')
        torch.save(self.model.state_dict(), cpfile)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def main():
    metadata = {}
    with open('parameters.json') as f:
        all_tests = json.load(f)
    for metadata in all_tests:
        config = metadata['config']
        args = metadata['args']
        device = metadata['args']['device']
        exp_name = metadata['args']['exp_name']
        # Training parameters
        batch_size = args["batch_size"]
        epochs = args["epochs"]
        lr = args["lr"]
        device = args["device"]
        save_model_every_n_epochs = args["save_model_every"]
        # Load the CIFAR10 dataset
        trainloader, testloader, _ = prepare_data(batch_size=batch_size)
        # Create the model, optimizer, loss function and trainer
        model = ViTForClassfication(config)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, loss_fn,
                          args["exp_name"], device=device, config=config)
        tic = time.perf_counter()
        trainer.train(trainloader, testloader, epochs,
                      save_model_every_n_epochs=save_model_every_n_epochs)
        toc = time.perf_counter()
        seconds_per_epoch = (toc - tic) / epochs
        outdir = f"./experiments/{exp_name}/"
        jsonfile = os.path.join(outdir, 'metrics.json')
        with open(jsonfile, 'r') as f:
            data = json.load(f)
        train_losses = data['train_losses']
        test_losses = data['test_losses']
        accuracies = data['accuracies']
        # Create two subplots of train/test losses and accuracies
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        plt.suptitle(f"{exp_name}, {seconds_per_epoch:0.4f} seconds per epoch")
        ax1.plot(train_losses, label="Train loss")
        ax1.plot(test_losses, label="Test loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.plot(accuracies)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        plt.savefig(f"metrics_{exp_name}.png")


if __name__ == "__main__":
    main()
