class Scheduler():
    def __init__(self, heb_lr, step_size, gamma=0.99):
        self.lr=heb_lr
        self.step_size=step_size
        self.gamma=gamma
        
        self.epoch=0
        
    def step(self):
        self.epoch+=1
        if (self.epoch % self.step_size)==0:
            self.lr=self.gamma*self.lr
        return self.lr
            