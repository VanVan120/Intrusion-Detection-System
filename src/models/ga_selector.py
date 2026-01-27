import time
import numpy as np
import pygad
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from .base_model import BaseModel

# Global variables required for PyGAD callback
# PyGAD functions must be picklable, so we often define them at module level
X_train_fold = None
X_val_fold = None
y_sub_train_global = None
y_sub_val_global = None
pbar_global = None
history_global = None

def calculate_fitness(ga_instance, solution, solution_idx):
    if X_train_fold is None: return 0 
    if sum(solution) == 0: return 0

    selected_indices = [i for i, x in enumerate(solution) if x == 1]
    
    # Accuracy
    X_train_sel = X_train_fold[:, selected_indices]
    X_val_sel = X_val_fold[:, selected_indices]
    
    # Use standard entropy to match baseline
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train_sel, y_sub_train_global)
    accuracy = clf.score(X_val_sel, y_sub_val_global)
    
    # Correlation
    if len(selected_indices) > 1:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_matrix = np.corrcoef(X_val_sel, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix)
        abs_corr = np.abs(corr_matrix)
        mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
        avg_corr = np.mean(abs_corr[mask]) if np.any(mask) else 0
    else:
        avg_corr = 0 

    # Fitness = (Acc + (1 - Corr)) / 2
    fitness = (accuracy + (1 - avg_corr)) / 2
    return fitness

def on_generation_callback(ga_instance):
    global pbar_global, history_global
    
    solution, fitness, _ = ga_instance.best_solution()
    selected_indices = [i for i, x in enumerate(solution) if x == 1]
    
    acc = 0.0
    if len(selected_indices) > 0:
        X_train_sel = X_train_fold[:, selected_indices]
        X_val_sel = X_val_fold[:, selected_indices]
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_train_sel, y_sub_train_global)
        acc = clf.score(X_val_sel, y_sub_val_global)

    history_global['iteration'].append(ga_instance.generations_completed)
    history_global['fitness'].append(fitness)
    history_global['accuracy'].append(acc)
    history_global['features'].append(len(selected_indices))
    
    if pbar_global:
        pbar_global.update(1)
        pbar_global.set_description(f"Gen {ga_instance.generations_completed} - Fit: {fitness:.4f} | Acc: {acc:.4f}")

class GASelector(BaseModel):
    def run(self):
        global X_train_fold, X_val_fold, y_sub_train_global, y_sub_val_global, pbar_global, history_global
        
        # Split data for internal fitness calculation
        X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # Set globals
        X_train_fold = X_sub_train
        X_val_fold = X_sub_val
        y_sub_train_global = y_sub_train
        y_sub_val_global = y_sub_val
        history_global = self.history
        
        num_gens = self.config['ga']['num_generations']
        pop_size = self.config['ga']['population_size']
        pbar_global = tqdm(total=num_gens, desc="GA Evolution")
        
        n_jobs = os.cpu_count()
        
        self.ga_instance = pygad.GA(
            num_generations=num_gens,
            num_parents_mating=int(pop_size / 2),
            fitness_func=calculate_fitness,
            sol_per_pop=pop_size,
            num_genes=self.X_train.shape[1],
            gene_space=[0, 1],
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=self.config['ga']['mutation_percent_genes'],
            suppress_warnings=True,
            on_generation=on_generation_callback,
            parallel_processing=["thread", n_jobs]
        )
        
        print(f"Starting GA Optimization...")
        start_time = time.time()
        try:
            self.ga_instance.run()
        except Exception as e:
            print(f"Parallel processing failed: {e}. Retry single-threaded.")
            self.ga_instance.parallel_processing = None
            self.ga_instance.run()
            
        self.training_time = time.time() - start_time
        if pbar_global: pbar_global.close()
        
    def get_best_features(self):
        solution, fitness, _ = self.ga_instance.best_solution()
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        selected_names = [self.feature_names[i] for i in selected_indices]
        return selected_indices, selected_names, fitness, self.training_time
