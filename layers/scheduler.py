from typing import (Any, Dict)

class Scheduler():
    def __init__(self, heb_lr, step_size, gamma=0.99):
        self.lr=heb_lr
        self.step_size=step_size
        self.gamma=gamma
        
        self.epoch=0
        self.last_lr = heb_lr
        
    def step(self):
        self.epoch+=1
        if (self.epoch % self.step_size)==0:
            self.lr=self.gamma*self.lr
            self.last_lr=self.lr
        return self.lr
    
    def get_last_lr(self):
        return self.last_lr
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
    
            