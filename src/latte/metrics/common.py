from abc import ABC, abstractmethod

class LatteMetric(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def add_state(self):
        pass
    
    @abstractmethod
    def update_state(self):
        pass
    
    @abstractmethod
    def reset_state(self):
        pass
    
    @abstractmethod
    def compute(self):
        pass