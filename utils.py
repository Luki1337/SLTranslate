import string
from PIL import Image
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
from PIL import Image
from torchvision import transforms
import random
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
           image_transformed = self.transform(image)
        else:
            image_transformed = transforms.ToTensor()(image)

        return image_transformed, torch.tensor(label, dtype=torch.long)

def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file

  Args:
    filename (string): path to the CSV file

  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:

    # Use csv.reader, passing in the appropriate delimiter
    # Remember that csv.reader can be iterated and returns one line in each iteration
    csv_reader = csv.reader(file, delimiter=',')


    # Skip the header
    next(csv_reader)

    # Initialize lists to store labels and images
    labels_list = []
    images_list = []

    # Process each row in the CSV file
    for row in csv_reader:
        # The first item is the label
        labels_list.append(int(row[0]))
        # The remaining items are the pixel values, reshaped from 784 to 28x28
        image = np.array(row[1:], dtype=np.float64).reshape(28, 28)
        images_list.append(image)

    # Convert lists to numpy arrays with type float64
    labels = np.array(labels_list, dtype=np.float64)
    images = np.array(images_list, dtype=np.float64)


    return images, labels

def plot_categories(training_images, training_labels):
    fig, axes = plt.subplots(1, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    for k in range(10):
        img = training_images[k]
        ax = axes[k]
        ax.imshow(img, cmap="gray")  # Use cmap="gray" for grayscale
        ax.set_title(f"{letters[int(training_labels[k])]}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def train(train_dataloader, valid_dataloader, model, criterion, optimizer, epochs, model_save_path, writer, device,
          scaler, scheduler, patience):
    """
    Train a neural network model and validate it on a separate validation set.

    Parameters:
    - train_dataset (torch.utils.data.Dataset): The training dataset.
    - valid_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - model (torch.nn.Module): The neural network model to be trained.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - epochs (int): The number of training epochs.
    - model_save_path (str): Path to save the best model.
    - writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
    - device (str): Device on which to perform training (e.g., 'cuda' or 'cpu').
    - scaler (torch.cuda.amp.GradScaler): GradScaler for mixed-precision training.
    - patience (int): Number of epochs with no improvement on the validation loss before stopping.

    Returns:
    None
    """
    # Initialize variables to track the best validation loss
    best_valid_loss = float('inf')
    no_improvement_count = 0  # Counter for consecutive epochs with no improvement

    # Print initial losses and accuracies
    print("Initial Stats: \n")
    initial_train_loss, initial_train_accuracy = evaluate_model(model, train_dataloader, criterion, device)
    initial_valid_loss, initial_valid_accuracy = evaluate_model(model, valid_dataloader, criterion, device)
    print(f'Train Loss: {initial_train_loss:.4f}, Train Accuracy: {initial_train_accuracy:.4f}')
    print(f'Validation Loss: {initial_valid_loss:.4f}, Validation Accuracy: {initial_valid_accuracy:.4f}')
    print("-" * 50)

    # Logging to TensorBoard
    writer.add_scalar('Loss/Train', initial_train_loss, 0)
    writer.add_scalar('Loss/Valid', initial_valid_loss, 0)
    writer.add_scalar('Accuracy/Train', initial_train_accuracy, 0)
    writer.add_scalar('Accuracy/Valid', initial_valid_accuracy, 0)

    print("\nStarting training: \n")
    # Training loop
    for epoch in range(1, epochs + 1):

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training", position=0, leave=False):
            # Move inputs to the specified device
            inputs = inputs.to(device=device, memory_format=torch.channels_last)
            labels = labels.squeeze().long().to(device)

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # get prediction and loss
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # use scaler to scale loss for mixed-precision training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Scheduler
            scheduler.step()

            # Accumulate the training loss
            train_loss += loss.item() * inputs.size(0)

            outputs = outputs.to("cpu")
            labels = labels.to("cpu")

            # Compute the number of correctly predicted samples
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Average the training loss over the dataset
        train_loss /= len(train_dataloader.dataset)

        # Compute training accuracy
        train_accuracy = train_correct / train_total

        # Use evaluate_model function for validation
        valid_loss, valid_accuracy = evaluate_model(model, valid_dataloader, criterion, device)

        # Scheduler step based on the validation loss (if scheduler is provided)
        if scheduler is not None:
            scheduler.step(valid_loss)

        # Save the best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            no_improvement_count = 0  # Reset the counter on improvement
        else:
            no_improvement_count += 1

        # Logging to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Valid', valid_accuracy, epoch)

        # Print stats every 5 epochs and the last epoch
        if epoch % 5 == 0:
            print(f'Epoch {epoch}/{epochs} ==> '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}',
                  f'Validation Loss: {valid_loss:.4f}, '
                  f'Validation Accuracy: {valid_accuracy:.4f}')

        # Check for early stopping
        if no_improvement_count >= patience:
            print(f'No improvement for {patience} consecutive epochs. Stopping training.')
            break

    # Close TensorBoard writer
    writer.close()


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader and return the loss and accuracy.

    Parameters:
    - model (torch.nn.Module): The neural network model to be evaluated.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - criterion (torch.nn.Module): The loss function.
    - device (str): Device on which to perform evaluation (e.g., 'cuda' or 'cpu').

    Returns:
    - loss (float): Average loss on the dataset.
    - accuracy (float): Accuracy on the dataset.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation", position=0, leave=False):
            inputs = inputs.to(device=device, memory_format=torch.channels_last)
            labels = labels.squeeze().long().to(device)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(dataloader.dataset)
    accuracy = correct / total
    return loss, accuracy
