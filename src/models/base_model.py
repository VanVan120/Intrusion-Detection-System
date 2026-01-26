from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, X_train, y_train, feature_names, config):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.config = config
        self.history = {'iteration': [], 'fitness': [], 'accuracy': [], 'features': []}
        self.training_time = 0
        
    @abstractmethod
    def run(self):
        """Execute the optimization process."""
        pass
    
    @abstractmethod
    def get_best_features(self):
        """Return the best selected feature indices and names."""
        pass
