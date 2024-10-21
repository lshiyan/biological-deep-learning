import torch
import torch.nn as nn
from tqdm.auto import tqdm
import models.learning as L
import models.helper_modules as M
from models.hyperparams import (LearningRule, WeightScale, oneHotEncode, InputProcessing,
                                Inhibition, WeightGrowth)
from dotwiz import DotWiz


class SoftHebbLayer(nn.Module):
    def __init__(self, inputdim: int, outputdim: int, w_lr: float = 0.003, b_lr: float = 0.003, l_lr: float = 0.003,
                 device=None, is_output_layer=False, initial_weight_norm: float = 0.01,
                 triangle:bool = True, initial_lambda: float = 4.0,
                 inhibition: Inhibition = Inhibition.Softmax,
                 learningrule: LearningRule = LearningRule.SoftHebb,
                 preprocessing: InputProcessing = InputProcessing.Whiten, # whitening or no whitening
                 weight_growth: WeightGrowth = WeightGrowth.Default):
        super(SoftHebbLayer, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_dim: int = inputdim
        self.output_dim: int = outputdim
        self.triangle: bool = triangle
        self.w_lr: float = w_lr
        self.l_lr: float = l_lr
        self.b_lr: float = b_lr
        self.lamb = nn.Parameter(torch.tensor(initial_lambda, device=device), requires_grad=False)
        self.is_output_layer: bool = is_output_layer
        self.learningrule: LearningRule = learningrule
        assert learningrule in [LearningRule.SoftHebb, LearningRule.SoftHebbOutputContrastive]
        self.inhibition: Inhibition = inhibition
        self.weight_growth: WeightGrowth = weight_growth
        self.preprocessing: InputProcessing = preprocessing
        if preprocessing == InputProcessing.Whiten:
            self.bn = M.BatchNorm(inputdim, device=device)

        self.weight = nn.Parameter(torch.randn((outputdim, inputdim), device=device), requires_grad=False)
        self.logprior = nn.Parameter(torch.zeros(outputdim, device=device), requires_grad=False)

        self.initial_weight_norm = initial_weight_norm
        self.set_weight_norms_to(initial_weight_norm)


    def set_weight_norms_to(self, norm: float):
        weights = self.weight
        weights_norm = self.get_weight_norms(weights)
        new_weights = norm * weights / (weights_norm + 1e-10)
        self.weight = nn.Parameter(new_weights, requires_grad=False)

    def get_weight_norms(self, weights):
        return torch.norm(weights, p=2, dim=1, keepdim=True)

    def a(self, x):
        # batch_size, dim = x.shape
        # it is expected that x has L2 norm of 1.
        weight_norms = self.get_weight_norms(self.weight)
        W = self.weight / (weight_norms + 1e-9)
        cos_sims = torch.matmul(x, W.T)
        return cos_sims

    def u(self, a):
        # batch_size, dim = a.shape
        if self.inhibition == Inhibition.RePU:
            if self.triangle:
                setpoint = a.mean()
            else:
                setpoint = 0
            u = torch.relu(a - setpoint)
        elif self.inhibition == Inhibition.Softmax:
            u = torch.exp(a)
        else:
            raise NotImplementedError(f"{self.inhibition} is not an implemented inhibition method.")
        return u

    def y(self, a):
        if self.inhibition == Inhibition.Softmax:
            y = torch.softmax(self.lamb * a + self.logprior, dim=1)    
        elif self.inhibition == Inhibition.RePU:
            u = self.u(a)
            un = u / (torch.max(u) + 1e-9)   # normalize for numerical stability
            ulamb = un ** self.lamb * torch.exp(self.logprior)
            y = ulamb / (torch.sum(ulamb, dim=1, keepdim=True) + 1e-9)
        else:
            raise NotImplementedError(f"{self.inhibition} is not an implemented inhibition method.")
        return y
    
    def inference(self, x):
        x_norms = torch.norm(x, dim=1, keepdim=True)
        x_n = x / (x_norms + 1e-9)
        a = self.a(x_n)
        u = self.u(a)
        y = self.y(a)
        return DotWiz(xn=x_n, a=a, u=u, y=y)

    def learn_weights(self, inference_output, target=None):
        supervised = self.learningrule == LearningRule.SoftHebbOutputContrastive
        delta_w = L.update_softhebb_w(inference_output.y, inference_output.xn, inference_output.a,
                                      self.weight, self.inhibition, inference_output.u, target=target,
                                      supervised=supervised, weight_growth=self.weight_growth)
        delta_b = L.update_softhebb_b(inference_output.y, self.logprior, target=target, supervised=supervised)
        delta_l = L.update_softhebb_lamb(inference_output.y, inference_output.a, inhibition=self.inhibition,
                                         lamb=self.lamb.item(), in_dim=self.input_dim, target=target,
                                         supervised=supervised)
        new_weight = self.weight + self.w_lr * delta_w
        self.weight.data = new_weight

        new_bias = self.logprior + self.b_lr * delta_b
        #normalize bias to be proper prior, i.e. sum exp Prior = 1
        norm_cste = torch.log(torch.exp(new_bias).sum())
        self.logprior.data = new_bias - norm_cste

        new_lambda = self.lamb + self.l_lr * delta_l
        self.lamb.data = new_lambda


    def forward(self, x, target=None):
        inference_output = self.inference(x)
        if self.training:
            self.learn_weights(inference_output, target=target)
        return inference_output.y

class Hebbian_Layer(nn.Module):
    def __init__(self, inputdim, outputdim, lr, lamb, w_decrease, gamma, eps, device, is_output_layer=False, 
                 output_learning=LearningRule.Supervised, update=LearningRule.FullyOrthogonal, weight=WeightScale.WeightDecay):
        super(Hebbian_Layer, self).__init__()
        self.input_dim = inputdim
        self.output_dim = outputdim
        self.lr = lr
        self.lamb = lamb
        self.decrease = w_decrease
        self.gamma = gamma
        self.epsilon = eps
        self.is_output_layer = is_output_layer

        self.feedforward = nn.Linear(self.input_dim, self.output_dim, bias=False)

        for param in self.feedforward.parameters():
            param=torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
        
        self.exponential_average=torch.zeros(self.output_dim).to(device)
        self.o_learning = output_learning
        self.w_update = update
        self.weight_mod = weight
        self.device = device


    def inhibition(self, x):
        x=nn.ReLU()(x)
        max_ele=torch.max(x).item()
        x/=max_ele
        x=torch.pow(x, self.lamb)
        return x
        
    # Fully orthogonal Sanger variant
    def update_weights_FullyOrthogonal(self, input, output):
        y = output.squeeze()
        weight = self.feedforward.weight
        delta_w = L.update_weights_FullyOrthogonal(input, output, weight)
        delta_w = torch.mean(delta_w, dim=0)
        delta_weight = self.lr*(delta_w)
        new_weights = torch.add(weight, delta_weight)
        self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weights_OrthogonalExclusive(self, input: torch.Tensor, output: torch.Tensor):
        y = output.squeeze()
        weight = self.feedforward.weight
        delta_w = L.update_weights_OrthogonalExclusive(input, output, weight, self.device)
        delta_w = torch.mean(delta_w, dim=0)
        computed_rule: torch.Tensor = self.lr*(delta_w)
        self.feedforward.weight=nn.Parameter(torch.add(computed_rule, weight), requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def update_weight_softhebb(self, input, preac, postac):
        y = postac.squeeze()
        weight = self.feedforward.weight 
        delta_w = L.update_weight_softhebb(input, preac, postac, weight, inhibition=self.inhib)
        delta_w = self.lr*delta_w
        self.feedforward.weight=nn.Parameter(torch.add(delta_w, weight), requires_grad=False)
        self.exponential_average=torch.add(self.gamma*self.exponential_average,(1-self.gamma)*y)
    
    def classifier_update_contrastive(self, input, output, true_output):
        output = output.squeeze()
        input = input.squeeze()
        true_output = true_output.squeeze()
        outer = torch.outer(nn.ReLU()(true_output - output), input)
        self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)

    def classifier_update_supervised(self, input, output):
        output = output.detach().clone().squeeze()
        input = input.detach().clone().squeeze()
        outer = torch.outer(output, input)
        self.feedforward.weight=nn.Parameter(torch.add(outer, self.feedforward.weight.detach().clone()), requires_grad=False)
    
    def normalize_weights(self):
        new_weights = self.feedforward.weight.detach().clone()
        #print(new_weights.shape)
        #print(new_weights)
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True)
        new_weights_norm = torch.norm(new_weights, p=2, dim=1, keepdim=True)
        #print(new_weights_norm)
        #print(new_weights_norm.shape)
        new_weights = new_weights / (new_weights_norm)
        new_weights = new_weights / (new_weights_norm)
        #print(new_weights)
        self.feedforward.weight=nn.Parameter(new_weights, requires_grad=False)
    
    def weight_decay(self):
        average=torch.mean(self.exponential_average).item()
        A=self.exponential_average/average
        growth_factor_positive=self.epsilon*nn.Tanh()(-self.epsilon*(A-1))+1
        growth_factor_negative=torch.reciprocal(growth_factor_positive)
        positive_weights=torch.where(self.feedforward.weight>0, self.feedforward.weight, 0.0)
        negative_weights=torch.where(self.feedforward.weight<0, self.feedforward.weight, 0.0)
        positive_weights=positive_weights*growth_factor_positive.unsqueeze(1)
        negative_weights=negative_weights*growth_factor_negative.unsqueeze(1)
        self.feedforward.weight=nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)

    def forward(self, x, clamped):
        x = x.reshape(-1).unsqueeze(0)
        input = x.detach().clone()
        #self.normalize_weights()
        h = self.feedforward(x)
        x = self.inhibition(h)
        if self.is_output_layer:
            if self.o_learning == LearningRule.OutputContrastiveSupervised:
                self.classifier_update_contrastive(input, x, clamped)
            elif self.o_learning == LearningRule.Supervised:
                self.classifier_update_supervised(input, clamped)
            else:
                self.update_weights_FullyOrthogonal(input, clamped)
        else:
            if self.w_update == LearningRule.FullyOrthogonal:
                self.update_weights_FullyOrthogonal(input, x)
            elif self.w_update == LearningRule.Softhebb:
                self.update_weight_softhebb(input, h, x)
            else:
                self.update_weights_OrthogonalExclusive(input, x)

            if self.weight_mod == WeightScale.WeightDecay:
                self.weight_decay()
            else:
                self.normalize_weights()
        return x
            
    def forward_test(self, x):
        x = x.reshape(-1).unsqueeze(0)
        x = self.feedforward(x)
        x = self.inhibition(x)
        return x
    
    def TD_forward(self, x, w, h_l):
        if h_l == None:
            h = self.feedforward(x)
        else :
            h = self.feedforward(x) + self.decrease*torch.matmul(h_l, w)
        return h, self.inhibition(h)