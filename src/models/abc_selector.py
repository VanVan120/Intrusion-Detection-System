import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from joblib import Parallel, delayed
from .base_model import BaseModel

class ABCSelector(BaseModel):
    def __init__(self, X_train, y_train, feature_names, config):
        super().__init__(X_train, y_train, feature_names, config)
        self.X_sub, self.X_val, self.y_sub, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        self.n_jobs = os.cpu_count()
        self.limit = self.config['abc']['limit']

    def calculate_fitness(self, solution):
        selected_indices = np.where(solution > 0.5)[0]
        if len(selected_indices) == 0:
            return 0.0
            
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        
        # Fitness = Acc + (1/Feat) small weight
        fitness = acc + 0.001 * (1 - (len(selected_indices)/len(solution)))
        return fitness

    def run(self):
        colony_size = self.config['abc']['colony_size']
        n_dim = self.X_train.shape[1]
        
        # Initialize
        foods = np.random.uniform(0, 1, size=(colony_size, n_dim))
        fitness = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(ind) for ind in foods)
        fitness = np.array(fitness)
        trial = np.zeros(colony_size)
        
        best_idx = np.argmax(fitness)
        self.best_solution = foods[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        
        print("Starting ABC Optimization...")
        start_time = time.time()
        
        for it in tqdm(range(self.config['abc']['n_iterations']), desc="ABC Evolution"):
            # 1. Employed Bees
            for i in range(colony_size):
                k = np.random.randint(colony_size)
                while k == i: k = np.random.randint(colony_size)
                j = np.random.randint(n_dim)
                
                phi = np.random.uniform(-1, 1)
                new_solution = foods[i].copy()
                new_solution[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                new_solution = np.clip(new_solution, 0, 1)
                
                new_fit = self.calculate_fitness(new_solution)
                
                if new_fit > fitness[i]:
                    foods[i] = new_solution
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # 2. Onlooker Bees
            prob = fitness / fitness.sum()
            t = 0
            i = 0
            while t < colony_size:
                if np.random.rand() < prob[i]:
                    t += 1
                    k = np.random.randint(colony_size)
                    while k == i: k = np.random.randint(colony_size)
                    j = np.random.randint(n_dim)
                    
                    phi = np.random.uniform(-1, 1)
                    new_solution = foods[i].copy()
                    new_solution[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                    new_solution = np.clip(new_solution, 0, 1)
                    
                    new_fit = self.calculate_fitness(new_solution)
                    if new_fit > fitness[i]:
                        foods[i] = new_solution
                        fitness[i] = new_fit
                        trial[i] = 0
                    else:
                        trial[i] += 1
                i = (i + 1) % colony_size
                
            # 3. Scout Bees
            max_trial_idx = np.argmax(trial)
            if trial[max_trial_idx] > self.limit:
                foods[max_trial_idx] = np.random.uniform(0, 1, size=n_dim)
                fitness[max_trial_idx] = self.calculate_fitness(foods[max_trial_idx])
                trial[max_trial_idx] = 0
                
            # Update Best
            curr_best_idx = np.argmax(fitness)
            if fitness[curr_best_idx] > self.best_fitness:
                self.best_solution = foods[curr_best_idx].copy()
                self.best_fitness = fitness[curr_best_idx]
                
            # Track
            self.history['iteration'].append(it)
            self.history['fitness'].append(self.best_fitness)
            
        self.training_time = time.time() - start_time
        
    def get_best_features(self):
        selected_indices = np.where(self.best_solution > 0.5)[0]
        selected_names = [self.feature_names[i] for i in selected_indices]
        return selected_indices, selected_names, self.best_fitness, self.training_time
