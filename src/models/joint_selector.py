import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from .base_model import BaseModel

class JointSelector(BaseModel):
    def __init__(self, X_train, y_train, feature_names, config):
        super().__init__(X_train, y_train, feature_names, config)
        
        params = config.get('joint', {})
        self.n_particles = params.get('n_particles', 20)
        self.n_iterations = params.get('n_iterations', 10)
        self.n_jobs = params.get('n_jobs', 4)
        self.alpha = params.get('alpha', 0.005)
        
        # PSO Params
        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.v_max = 0.2
        self.mutation_rate = 0.05
        
        # Dimensions
        self.n_features = X_train.shape[1]
        self.n_params = 3 # max_depth, min_samples_split, criterion
        self.total_dim = self.n_features + self.n_params
        
        # Internal validation split
        self.X_sub, self.X_val, self.y_sub, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # Initialize Population
        self.positions = np.random.uniform(0, 1, size=(self.n_particles, self.total_dim))
        self.velocities = np.random.uniform(-self.v_max, self.v_max, size=(self.n_particles, self.total_dim))
        
        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, -1.0)
        self.gbest_pos = None
        self.gbest_score = -1.0
        
        # Store best found config
        self.best_hyperparams = {}

    def decode_particle(self, particle):
        # Feature Segment
        feature_vals = particle[:self.n_features]
        selected_indices = np.where(feature_vals > 0.5)[0]
        
        # Parameter Segment
        param_vals = particle[self.n_features:]
        
        # Param 1: max_depth [1, 30]
        p0 = np.clip(param_vals[0], 0, 0.9999)
        max_depth = 1 + int(p0 * 29)
        
        # Param 2: min_samples_split [2, 20]
        p1 = np.clip(param_vals[1], 0, 0.9999)
        min_split = 2 + int(p1 * 18)
        
        # Param 3: criterion {gini, entropy}
        p2 = np.clip(param_vals[2], 0, 0.9999)
        criterion = 'entropy' if p2 > 0.5 else 'gini'
        
        return selected_indices, {'max_depth': max_depth, 
                                  'min_samples_split': min_split, 
                                  'criterion': criterion}

    def calculate_fitness(self, particle):
        selected_indices, params = self.decode_particle(particle)
        
        if len(selected_indices) == 0:
            return 0.0
            
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            criterion=params['criterion'],
            random_state=42
        )
        clf.fit(X_sub_sel, self.y_sub)
        pred = clf.predict(X_val_sel)
        
        acc = accuracy_score(self.y_val, pred)
        feat_ratio = len(selected_indices) / self.n_features
        fitness = acc + self.alpha * (1 - feat_ratio)
        
        return fitness

    def calculate_metrics_for_history(self, particle):
        # Helper to get exact accuracy for history logging
        selected_indices, params = self.decode_particle(particle)
        if len(selected_indices) == 0: return 0.0, 0
        
        X_sub_sel = self.X_sub[:, selected_indices]
        X_val_sel = self.X_val[:, selected_indices]
        
        clf = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            criterion=params['criterion'],
            random_state=42
        )
        clf.fit(X_sub_sel, self.y_sub)
        return accuracy_score(self.y_val, clf.predict(X_val_sel)), len(selected_indices)

    def run(self):
        print(f"Starting Joint Optimization (Features + Hyperparams)...")
        start_time = time.time()
        
        # Initial Eval
        scores = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(p) for p in self.positions)
        scores = np.array(scores)
        
        self.pbest_scores = scores.copy()
        best_idx = np.argmax(scores)
        self.gbest_pos = self.positions[best_idx].copy()
        self.gbest_score = scores[best_idx]
        
        for iteration in range(self.n_iterations):
            # PSO Update
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_pos[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.gbest_pos - self.positions[i]))
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], 0, 1)
            
            # Mutation (Simple GA on non-best)
            non_best_masks = [not np.array_equal(p, self.gbest_pos) for p in self.positions]
            non_best_indices = np.where(non_best_masks)[0]
            if len(non_best_indices) > 0:
                n_mutants = max(1, int(len(non_best_indices) * 0.2))
                mutant_indices = np.random.choice(non_best_indices, n_mutants, replace=False)
                for idx in mutant_indices:
                    mutation_mask = np.random.rand(self.total_dim) < self.mutation_rate
                    # Mutate by re-initializing random dimensions
                    self.positions[idx][mutation_mask] = np.random.rand(np.sum(mutation_mask))

            # Evaluate
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.calculate_fitness)(p) for p in self.positions)
            scores = np.array(scores)
            
            # Update Personal Best
            better_mask = scores > self.pbest_scores
            self.pbest_pos[better_mask] = self.positions[better_mask]
            self.pbest_scores[better_mask] = scores[better_mask]
            
            # Update Global Best
            current_best_idx = np.argmax(scores)
            if scores[current_best_idx] > self.gbest_score:
                self.gbest_score = scores[current_best_idx]
                self.gbest_pos = self.positions[current_best_idx].copy()
            
            # Log history
            best_acc, best_feat_count = self.calculate_metrics_for_history(self.gbest_pos)
            self.history['iteration'].append(iteration)
            self.history['fitness'].append(self.gbest_score)
            self.history['accuracy'].append(best_acc)
            self.history['features'].append(best_feat_count)
            
            print(f"Iter {iteration+1}/{self.n_iterations} | Best Fitness: {self.gbest_score:.4f} | Acc: {best_acc:.4f} | Feats: {best_feat_count}")
        
        self.training_time = time.time() - start_time
        # Save best hyperparams for retrieval
        _, self.best_hyperparams = self.decode_particle(self.gbest_pos)

    def get_best_features(self):
        best_indices, _ = self.decode_particle(self.gbest_pos)
        feature_names_selected = [self.feature_names[i] for i in best_indices]
        return best_indices, feature_names_selected, self.history, self.training_time

    def get_best_hyperparams(self):
        return self.best_hyperparams
