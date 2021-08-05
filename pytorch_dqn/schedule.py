class PiecewiseLinearEpsilon():
    def __init__(self, frames=[0,50000,100000,1000000], epsilons=[1.0, 1.0, 0.1, 0.01]):
        assert len(frames) == len(epsilons), "frames and epsilons must have the same length."
        
        self.frames=frames
        self.epsilons=epsilons
        
    def get_epsilon(self,t):
        for k in range(1,len(self.epsilons)):
            if t >= self.frames[k-1] and t < self.frames[k]:
                slope = (self.epsilons[k] - self.epsilons[k-1])/(self.frames[k] - self.frames[k-1])
                epsilon = self.epsilons[k-1] + slope*(t-self.frames[k-1])
                return epsilon
            
        return self.epsilons[-1]