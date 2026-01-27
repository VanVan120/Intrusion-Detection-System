import time
import numpy as np
import pyswarms as ps
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from joblib import Parallel, delayed
from .base_model import BaseModel

class PSOSelector(BaseModel):
    def __init__(self, X_train, y_train, feature_names, config):
        super().__init__(X_train, y_train, feature_names, config)
        self.X_sub, self.X_val, self.y_sub, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        self.n_jobs = os.cpu_count()

    def _evaluate_metric(self, m):
        """Helper to calculate accuracy for a specific particle needed for history tracking."""
        if np.count_nonzero(m) == 0:
            return 0.0, 0
            
        X_sub_sel = self.X_sub[:, m == 1]
        X_val_sel = self.X_val[:, m == 1]
        
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        return acc, np.count_nonzero(m)

    def f_per_particle(self, m, alpha=0.9):
        """Computes fitness for a single particle."""
        if np.count_nonzero(m) == 0:
            return 1.0  # Max penalty if no features selected
        
        X_sub_sel = self.X_sub[:, m == 1]
        X_val_sel = self.X_val[:, m == 1]
        
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_sub_sel, self.y_sub)
        acc = clf.score(X_val_sel, self.y_val)
        
        # Fitness: Weighted sum of error rate and feature ratio
        # Minimize J = alpha * (1 - Acc) + (1 - alpha) * (feat / total)
        # Corrected: We want to Minimize Feature Count (Ratio), not (1-Ratio)
        j = (alpha * (1.0 - acc)) + ((1.0 - alpha) * (X_sub_sel.shape[1] / self.X_sub.shape[1]))
        return j

    def f(self, x, alpha=0.9):
        """Computes fitness for the whole swarm."""
        n_particles = x.shape[0]
        # Pyswarms provides continuous values, threshold at 0.5
        m = (x > 0.5)
        
        # Parallel evaluation of Cost
        j = Parallel(n_jobs=self.n_jobs)(delayed(self.f_per_particle)(m[i], alpha) for i in range(n_particles))
        j = np.array(j)
        
        # Capture history (Best of this iteration)
        best_idx = np.argmin(j)
        best_m = m[best_idx]
        
        # We re-evaluate metrics for the best particle to store exact accuracy/features 
        # (small overhead: 1 extra training per generation)
        acc, feat_count = self._evaluate_metric(best_m)
        
        self.history['accuracy'].append(acc)
        self.history['features'].append(feat_count)
        self.history['fitness'].append(j[best_idx])
        iter_num = len(self.history['iteration']) + 1
        self.history['iteration'].append(iter_num)
        
        print(f"Gen {iter_num} - Best Fit: {j[best_idx]:.4f} | Acc: {acc:.4f} | Feat: {feat_count}")
        
        return j

    def run(self):
        options = {
            'c1': self.config['pso']['c1'], 
            'c2': self.config['pso']['c2'], 
            'w': self.config['pso']['w'],
            'k': self.config['pso'].get('k', 5),
            'p': self.config['pso'].get('p', 2)
        }
        dims = self.X_train.shape[1]
        
        # Initialize swarm
        optimizer = ps.discrete.BinaryPSO(
            n_particles=self.config['pso']['n_particles'],
            dimensions=dims,
            options=options
        )

        print("Starting PSO Optimization...")
        start_time = time.time()
        
        # Perform optimization
        cost, pos = optimizer.optimize(
            self.f, 
            iters=self.config['pso']['n_iterations'],
            verbose=False
        )
        
        self.training_time = time.time() - start_time
        self.best_pos = pos
        self.best_cost = cost
        
        # History is now populated inside f()
        
    def get_best_features(self):
        selected_indices = [i for i, x in enumerate(self.best_pos) if x == 1]
        selected_names = [self.feature_names[i] for i in selected_indices]
        return selected_indices, selected_names, self.best_cost, self.training_time
