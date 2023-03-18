import torch
from torch import nn

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

import math
assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertion to use CPU)"

def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
    )-> Tuple[List[float], List[float], List[float], List[float]]:
  """
  Trains a model for the specified number of epochs using the loaders.

  Returns: 
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
  """

  loss = nn.CrossEntropyLoss()
  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = []
  for e in range(epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    print(e)
    # Main training loop; iterate over train_loader. The loop
    # terminates when the train loader finishes iterating, which is one epoch.
    for (x_batch, labels) in train_loader:
      x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      labels_pred = model(x_batch)
      batch_loss = loss(labels_pred, labels)
      train_loss = train_loss + batch_loss.item()

      labels_pred_max = torch.argmax(labels_pred, 1)
      batch_acc = torch.sum(labels_pred_max == labels)
      train_acc = train_acc + batch_acc.item()
    
      batch_loss.backward()
      optimizer.step()
      
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_acc / (batch_size * len(train_loader)))

    # Validation loop; use .no_grad() context manager to save memory.
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
      for (v_batch, labels) in val_loader:
        v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
        labels_pred = model(v_batch)
        v_batch_loss = loss(labels_pred, labels)
        val_loss = val_loss + v_batch_loss.item()

        v_pred_max = torch.argmax(labels_pred, 1)
        batch_acc = torch.sum(v_pred_max == labels)
        val_acc = val_acc + batch_acc.item()
      val_losses.append(val_loss / len(val_loader))
      val_accuracies.append(val_acc / (batch_size * len(val_loader)))

  return train_losses, train_accuracies, val_losses, val_accuracies

def parameter_search(train_loader: DataLoader, 
                     val_loader: DataLoader, 
                     model_fn:Callable[[], nn.Module]) -> float:
  """
  Parameter search for our linear model using SGD.

  Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

  Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
  """
  #num_iter = 10  # This will likely not be enough for the rest of the problem.
  best_loss = float('inf')
  best_lr = 0.0

  #lrs = torch.tensor([10 ** -3, 0.005, 10 ** -2, 10 ** -1])
  lrs = torch.tensor([0.005, 10 ** -2, 10 ** -1])

  for lr in lrs:
    print(f"trying learning rate {lr}")
    model = model_fn()
    optim = SGD(model.parameters(), lr, momentum=0.99, weight_decay=weight_decay)

    train_loss, train_acc, val_loss, val_acc = train(
        model,
        optim,
        train_loader,
        val_loader,
        epochs=epoch_count
        )
    
    if min(val_loss) < best_loss:
      best_loss = min(val_loss)
      best_lr = lr

  print(best_lr)
  return best_lr

def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
  """Computes test loss and accuracy of model on loader."""
  loss = nn.CrossEntropyLoss()
  model.eval()
  test_loss = 0.0
  test_acc = 0.0
  with torch.no_grad():
    for (batch, labels) in loader:
      batch, labels = batch.to(DEVICE), labels.to(DEVICE)
      y_batch_pred = model(batch)
      batch_loss = loss(y_batch_pred, labels)
      test_loss = test_loss + batch_loss.item()

      pred_max = torch.argmax(y_batch_pred, 1)
      batch_acc = torch.sum(pred_max == labels)
      test_acc = test_acc + batch_acc.item()
    test_loss = test_loss / len(loader)
    test_acc = test_acc / (batch_size * len(loader))
    return test_loss, test_acc

def bird_classifer_model() -> nn.Module:
  model = nn.Sequential(
      nn.Conv2d(3, out, 3, padding=1, bias=False),
      nn.BatchNorm2d(out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(out, 2 * out, 3, padding=1, bias=False),
      nn.BatchNorm2d(2 * out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(2 * out, 4 * out, 3, padding=1, bias=False),
      nn.BatchNorm2d(4 * out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(4 * out, 8 * out, 3, padding=1, bias=False),
      nn.BatchNorm2d(8 * out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(8 * out, 16 * out, 3, padding=1, bias=False),
      nn.BatchNorm2d(16 * out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(16 * out, 32 * out, 3, padding=1, bias=False),
      nn.BatchNorm2d(32 * out),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(32 * out, num_labels)
  )
  return model.to(DEVICE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

# variables to tweak
batch_size = 128
new_img_size = 128
out = 8 # will range aross values
weight_decay = 0  # will range across values
percent_of_training_data_used = 0.3
percent_of_training_data_used_during_top_k = 0.11

# configure training, validation and test datasets
transform_train = transforms.Compose([
        transforms.Resize((new_img_size, new_img_size)),
        transforms.RandomCrop((new_img_size, new_img_size), padding=8, padding_mode='edge'), # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
])

transform_validation = transforms.Compose([
    transforms.Resize((new_img_size, new_img_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((new_img_size, new_img_size)),
    transforms.ToTensor(),
])

# get training dataset
train_dataset = torchvision.datasets.ImageFolder(root='./homeworks/train', transform=transform_train)
print(int(0.5 * len(train_dataset)),  len(train_dataset))
      
# only use a specified percentage of the training dataset
split_lengths = [int(percent_of_training_data_used * len(train_dataset)), int((1-percent_of_training_data_used) * len(train_dataset)), len(train_dataset) % 2]
split_lengths2 = [int(percent_of_training_data_used_during_top_k * len(train_dataset)), int((1-percent_of_training_data_used_during_top_k) * len(train_dataset)), len(train_dataset) % 2]
train1_dataset, train2_dataset, train3_dataset = random_split(train_dataset, split_lengths)
train_tune_1_dataset, train_tune_2_dataset, train_tune_3_dataset = random_split(train_dataset, split_lengths2)

# load training data
train_loader = torch.utils.data.DataLoader(train1_dataset, batch_size=batch_size, shuffle=True)
train_tune_loader = torch.utils.data.DataLoader(train_tune_1_dataset, batch_size=batch_size, shuffle=True)

# get and load validation data
validation_set = torchvision.datasets.ImageFolder(root='./homeworks/valid', transform=transform_validation)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# get and load test data
test_set = torchvision.datasets.ImageFolder(root='./homeworks/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

num_labels = len(train1_dataset.dataset.classes)
print(len(train1_dataset.dataset))
print(len(validation_set))
print(len(test_set))

imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
print(f"A single RGB image has {c} channels, width {w}, and height {h}.")

# This is one way to flatten our images
batch_flat_view = imgs.view(-1, c * w * h)
print(f"Size of a batch of images flattened with view: {batch_flat_view.size()}")

# This is another equivalent way
batch_flat_flatten = imgs.flatten(1)
print(f"Size of a batch of images flattened with flatten: {batch_flat_flatten.size()}")

# The new dimension is just the product of the ones we flattened
d = example_image.flatten().size()[0]
print(c * w * h == d)

#------------------------------------------------------------------------------------------------------------------------------------
# M hyperparameter search begins here
epoch_count = 6
out_possibilities = (torch.rand(3) * 10 + 1).int()
weight_possibilities = torch.tensor([1e-4, 1e-5])

val_accuracies = torch.zeros([out_possibilities.size(dim = 0), weight_possibilities.size(dim = 0), epoch_count])
train_accuracies = torch.zeros([out_possibilities.size(dim = 0), weight_possibilities.size(dim = 0), epoch_count])
best_lrs = torch.zeros([out_possibilities.size(dim = 0), weight_possibilities.size(dim = 0)])

test_accuracies = torch.zeros(3)


# loop through possible weights for regularization
for i in range(out_possibilities.size(dim = 0)):
    out = out_possibilities[i].item()
    # loop through possible numbers of output nodes on the first layer
    for j in range(weight_possibilities.size(dim = 0)):
      weight_decay = weight_possibilities[j].item()
    
      best_lrs[i][j] = parameter_search(train_tune_loader, validation_loader, bird_classifer_model)
      model = bird_classifer_model()
      optimizer = SGD(model.parameters(), best_lrs[i][j], momentum = 0.99, weight_decay=weight_decay)

      train_loss, train_accuracy, val_loss, val_accuracy = train(
      model, optimizer, train_tune_loader, validation_loader, epoch_count)
      
      train_accuracies[i][j] = torch.tensor(train_accuracy)
      val_accuracies[i][j] = torch.tensor(val_accuracy)
      print(out)
      print(weight_decay)
      print(train_accuracies[i][j])
      print(val_accuracies[i][j])

all_val_accuracies = val_accuracies[:,:, -1]
top_val_accuracies, top_indices = torch.topk(all_val_accuracies.view(-1), 3)

# use top 3 hyperparameter configuratons to train model on test data
epochs = range(1, epoch_count * 7 + 1)
for i in range(3):
   row = (top_indices[i] // 3)
   col = (top_indices[i] % 2)
   
   out = out_possibilities[row].item()
   weight_decay = weight_possibilities[col].item()
   
   print("best out: ", out)
   print("best weight decay: ", weight_decay)
   print("best lr: ", best_lrs[row][col])
   
   model = bird_classifer_model()
   optimizer = SGD(model.parameters(), lr=best_lrs[row][col], momentum=0.99, weight_decay=weight_decay)
   
   train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, validation_loader, epoch_count * 7)
   
   test_loss, test_accuracy = evaluate(model, test_loader)
   print("Best test accuracy: ", test_accuracy)
   
   plt.plot(epochs, train_accuracy, label="Train Accuracy",  linestyle='solid')
   plt.plot(epochs, val_accuracy, label="Validation Accuracy", linestyle='dotted')
   plt.hlines(0.5, 1, epoch_count * 7, linestyle='dashdot')
   
   plt.xlabel("Epoch")
   plt.ylabel("Accuracy")
   plt.legend()
   plt.title("A Bird Classifier Convolutional Neural Network")
   plt.show()