import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from numpy import outer
from layers.layer import NetworkLayer

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionMechanism, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output

class HebbianLayerWithAttention(NetworkLayer):
    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super().__init__(input_dimension, output_dimension, device_id, lamb, heb_lr, eps)

        self.fc = nn.Linear(input_dimension, output_dimension, bias=False)
        self.attention = AttentionMechanism(output_dimension, output_dimension)
        self.exponential_average = torch.zeros(output_dimension)
        self.id_tensor = self.create_id_tensors()
        self.gamma = gamma

    def inhibition(self, x):
        relu = nn.ReLU()
        x = relu(x)
        max_ele = torch.max(x).item()
        x = torch.pow(x, self.lamb)
        x /= abs(max_ele) ** self.lamb
        return x

    def update_weights(self, input, output):
        x = input.clone().detach().float().squeeze().to(self.device_id)
        x.requires_grad_(False)
        y = output.clone().detach().float().squeeze().to(self.device_id)
        y.requires_grad_(False)

        outer_prod = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device_id)
        initial_weight = torch.transpose(self.fc.weight.clone().detach().to(self.device_id), 0, 1)
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)

        A = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))
        delta_weight = self.alpha * (outer_prod - A)
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)

    def update_bias(self, output):
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1 * self.fc.bias)  # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A / bias_maxes.item(), requires_grad=False)

    def weight_decay(self):
        tanh = nn.Tanh()
        average = torch.mean(self.exponential_average).item()
        A = self.exponential_average / average
        growth_factor_positive = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative = torch.reciprocal(growth_factor_positive)
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        if self.fc.weight.isnan().any():
            print("NAN WEIGHT")

    def forward(self, x):
        input_copy = x.clone().to(self.device_id).float()
        
        # Ensure the input dimension matches what is expected
        if input_copy.size(1) != self.input_dimension:
            raise ValueError(f"Expected input dimension of {self.input_dimension}, but got {input_copy.size(1)}")

        x = self.fc(input_copy)  # Apply the fully connected layer first
        x = self.attention(x)  # Then apply the attention mechanism
        
        x = self.inhibition(x)  # Apply inhibition
        self.update_weights(input_copy, x)  # Update weights
        self.weight_decay()  # Apply weight decay
        return x

    def visualize_weights(self, result_path):
        # Find value for row and column
        row = 0
        col = 0

        root = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Get the weights and create heatmap
        weight = self.fc.weight
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row * col):  
            random_feature_selector = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
            
            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device_id)
        
        file_path = result_path + '/hebbianlayerweights.png'
        plt.tight_layout()
        plt.savefig(file_path)

    def active_weights(self, beta):
        pass

    def create_id_tensors(self):
        return torch.eye(self.output_dimension).unsqueeze(0).repeat(self.output_dimension, 1, 1).to(self.device_id)