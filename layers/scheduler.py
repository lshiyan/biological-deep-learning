"""
Class to manage and update learning rate of optimization process
"""
class Scheduler():
    """
    Contructor method
    @param
        lr (float) = learing rate
        step_size (int) = step size -> how often the learning rate should be updated
        gamma (float) = decay factor -> factor to decay learning rate
    @attr.
        flag (str) = a string indicating wether the data is used for training or testing
        data_frame (list-like object) = dataset for training/testing
        labels (torch.Tensor) = labels of dataset
        epoch (int) = epoch counter
    """
    def __init__(self, lr, step_size, gamma=0.99):
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    """
    Method used to increment epoch counter and update learning rate at every multiple of step size
    """    
    def step(self):
        self.epoch += 1
        if (self.epoch % self.step_size) == 0:
            self.lr *= self.gamma
        return self.lr