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

    def _evaluate_metric(self, solution):
        """Helper to calculate accuracy/features for history tracking."""
        selected_indices = np.where(solution > 0.5)[0]
        if len(selected_indices) == 0:
            return 0.0, 0
            
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        return acc, len(selected_indices)

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
        
        pbar = tqdm(range(self.config['abc']['n_iterations']), desc="ABC Evolution")
        for it in pbar:
            # 1. Employed Bees (Parallelized Phase)
            candidates = []
            for i in range(colony_size):
                k = np.random.randint(colony_size)
                while k == i: k = np.random.randint(colony_size)
                j = np.random.randint(n_dim)
                
                phi = np.random.uniform(-1, 1)
                new_solution = foods[i].copy()
                new_solution[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                new_solution = np.clip(new_solution, 0, 1)
                candidates.append(new_solution)
            
            # Evaluate all employed candidates in parallel
            new_fits = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(c) for c in candidates)
            
            for i in range(colony_size):
                if new_fits[i] > fitness[i]:
                    foods[i] = candidates[i]
                    fitness[i] = new_fits[i]
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # 2. Onlooker Bees (Parallelized Phase)
            prob = fitness / fitness.sum()
            # Select food sources probabilistically (some might be selected multiple times)
            selected_indices = np.random.choice(colony_size, colony_size, p=prob)
            
            onlooker_cands = []
            for i in selected_indices:
                k = np.random.randint(colony_size)
                while k == i: k = np.random.randint(colony_size)
                j = np.random.randint(n_dim)
                
                phi = np.random.uniform(-1, 1)
                new_solution = foods[i].copy()
                new_solution[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                new_solution = np.clip(new_solution, 0, 1)
                onlooker_cands.append(new_solution)
                
            # Evaluate all onlooker candidates in parallel
            on_fits = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(c) for c in onlooker_cands)
            
            for idx, i_source in enumerate(selected_indices):
                if on_fits[idx] > fitness[i_source]:
                    foods[i_source] = onlooker_cands[idx]
                    fitness[i_source] = on_fits[idx]
                    trial[i_source] = 0
                else:
                    trial[i_source] += 1
                
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
            best_acc, best_feat = self._evaluate_metric(self.best_solution)
            self.history['iteration'].append(it)
            self.history['fitness'].append(self.best_fitness)
            self.history['accuracy'].append(best_acc)
            self.history['features'].append(best_feat)
            
            pbar.set_description(f"Gen {it+1} - Fit: {self.best_fitness:.4f} | Acc: {best_acc:.4f}")
            
        self.training_time = time.time() - start_time
        
    def get_best_features(self):
        selected_indices = np.where(self.best_solution > 0.5)[0]
        selected_names = [self.feature_names[i] for i in selected_indices]
        return selected_indices, selected_names, self.best_fitness, self.training_time
