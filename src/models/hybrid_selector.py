import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from joblib import Parallel, delayed
from .base_model import BaseModel

class HybridSelector(BaseModel):
    def __init__(self, X_train, y_train, feature_names, config):
        super().__init__(X_train, y_train, feature_names, config)
        self.X_sub, self.X_val, self.y_sub, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        self.n_jobs = os.cpu_count()
        self.params = self.config['hybrid']

    def calculate_fitness(self, particle):
        selected_indices = np.where(particle > 0.5)[0]
        if len(selected_indices) == 0:
            return 0.0
            
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        
        # Fitness
        fitness = acc + 0.001 * (1 - (len(selected_indices)/len(particle)))
        return fitness

    def _evaluate_metric(self, particle):
        """Helper to calculate accuracy/features for history tracking."""
        selected_indices = np.where(particle > 0.5)[0]
        if len(selected_indices) == 0:
            return 0.0, 0
            
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        return acc, len(selected_indices)

    def run(self):
        n_particles = self.params['n_particles']
        n_iterations = self.params['n_iterations']
        n_dim = self.X_train.shape[1]
        
        # Initialize
        positions = np.random.uniform(0, 1, size=(n_particles, n_dim))
        velocities = np.random.uniform(-0.2, 0.2, size=(n_particles, n_dim))
        
        pbest_pos = positions.copy()
        pbest_scores = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(p) for p in positions)
        pbest_scores = np.array(pbest_scores)
        
        gbest_idx = np.argmax(pbest_scores)
        gbest_pos = positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        print("Starting Hybrid PSO-GA Optimization...")
        start_time = time.time()
        
        pbar = tqdm(range(n_iterations), desc="Hybrid Evolution")
        for it in pbar:
            # 1. PSO Update
            for i in range(n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.params['w'] * velocities[i] + 
                                 self.params['c1'] * r1 * (pbest_pos[i] - positions[i]) + 
                                 self.params['c2'] * r2 * (gbest_pos - positions[i]))
                velocities[i] = np.clip(velocities[i], -0.2, 0.2)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                
            # 2. GA Mutation (Hybrid)
            # Find non-best particles
            for i in range(n_particles):
                if i != gbest_idx and np.random.rand() < 0.2: # 20% chance to mutate
                    mask = np.random.rand(n_dim) < 0.05 # 5% genes
                    positions[i][mask] = np.random.rand(np.sum(mask))

            # Evaluate
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(p) for p in positions)
            scores = np.array(scores)
            
            # Update Bests
            for i in range(n_particles):
                if scores[i] > pbest_scores[i]:
                    pbest_scores[i] = scores[i]
                    pbest_pos[i] = positions[i].copy()
                    
                if scores[i] > gbest_score:
                    gbest_score = scores[i]
                    gbest_pos = positions[i].copy()
                    gbest_idx = i # Update index
            
            # Record
            best_acc, best_feat = self._evaluate_metric(gbest_pos)
            self.history['iteration'].append(it)
            self.history['fitness'].append(gbest_score)
            self.history['accuracy'].append(best_acc)
            self.history['features'].append(best_feat)
            
            pbar.set_description(f"Gen {it+1} - Fit: {gbest_score:.4f} | Acc: {best_acc:.4f}")
            
        self.training_time = time.time() - start_time
        self.best_pos = gbest_pos
        self.best_score = gbest_score

    def get_best_features(self):
        selected_indices = np.where(self.best_pos > 0.5)[0]
        selected_names = [self.feature_names[i] for i in selected_indices]
        return selected_indices, selected_names, self.best_score, self.training_time
