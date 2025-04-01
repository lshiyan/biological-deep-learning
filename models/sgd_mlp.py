import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim


class SGDNetwork(nn.Module):
    """
    Defines a network that uses Stochastic Gradient Descent (SGD) for training.
    """
    def __init__(self, name: str, args: argparse.Namespace) -> None:
        super(SGDNetwork, self).__init__()

        self.input_dim: int = args.input_dim
        self.hidden_dim: int = args.hidden_dim
        self.output_dim: int = args.output_dim
        self.lr: float = args.lr
        self.activation: str = args.activation

        # Layers of the network
        self.hidden_layer = nn.Linear(self.input_dim, self.hidden_dim)  # Hidden layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)  # Output layer

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def update_weights(self, logits, target):
        # Training mode: backpropagate and update weights
        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, target)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
        loss_value = loss.item()  # Return loss for tracking during training
        return loss_value

    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor, 
                reconstruct: bool = False, freeze: bool = False) -> torch.Tensor:
        """
        Forward method for training and reconstruction.
        `clamped_output` is required for training to compute the loss.
        """
        # Flatten the input if necessary
        input = input.view(input.size(0), -1)  # Flatten input to (batch_size, 28*28)
            # Forward pass through the hidden layer
        input = input.to(next(self.parameters()).device)
        h = self.relu(self.hidden_layer(input))
        logits = self.output_layer(x)
        loss_value = self.update_weights(logits, clamped_output)

        return logits