import argparse
from pathlib import Path
import yaml
from src.data_loader import IDSDataLoader
from src.utils import evaluate_model, plot_pareto_front
from src.analysis import run_analysis
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'results'
PLOTS_DIR = PROJECT_ROOT / 'results' / 'plots'

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_optimizer_class(method):
    try:
        if method == 'ga':
            from src.models.ga_selector import GASelector
            return GASelector
        if method == 'pso':
            from src.models.pso_selector import PSOSelector
            return PSOSelector
        if method == 'abc':
            from src.models.abc_selector import ABCSelector
            return ABCSelector
        if method == 'hybrid':
            from src.models.hybrid_selector import HybridSelector
            return HybridSelector
        if method == 'joint':
            from src.models.joint_selector import JointSelector
            return JointSelector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{exc.name}' required for '{method}' optimization. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    raise ValueError(f"Method {method} not supported.")

def run_feature_selection(method, config, df_split, show_plots=True):
    X_train, X_test, y_train, y_test, feature_names = df_split

    if method not in ['ga', 'pso', 'abc', 'hybrid', 'joint']:
        raise ValueError("Method not supported. Choose from ['ga', 'pso', 'abc', 'hybrid', 'joint']")
        
    print(f"\n--- Running {method.upper()} Optimization ---")
    optimizer_class = get_optimizer_class(method)
    optimizer = optimizer_class(X_train, y_train, feature_names, config)
    optimizer.run()
    
    best_indices, best_names, best_fit, runtime = optimizer.get_best_features()
    
    print(f"\nOptimization Complete.")
    print(f"Time: {runtime:.2f}s")
    print(f"Selected {len(best_names)} features.")
    
    # Check for empty feature selection
    if len(best_indices) == 0:
        print("WARNING: No features were selected! Falling back to all features.")
        best_indices = list(range(X_train.shape[1]))

    # Train Final Model
    print("Training final model on selected features...")
    X_train_sel = X_train[:, best_indices]
    X_test_sel = X_test[:, best_indices]
    
    # Check if method gives specific hyperparameters (for Joint Opt)
    if hasattr(optimizer, 'get_best_hyperparams'):
        best_params = optimizer.get_best_hyperparams()
        print(f"Using optimized hyperparameters: {best_params}")
        clf = DecisionTreeClassifier(
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=config['model']['decision_tree']['random_state']
        )
    else:
        clf = DecisionTreeClassifier(criterion=config['model']['decision_tree']['criterion'], 
                                     random_state=config['model']['decision_tree']['random_state'])
    
    clf.fit(X_train_sel, y_train)
    
    # Evaluate
    metrics, y_pred = evaluate_model(clf, X_test_sel, y_test, len(best_indices), runtime, method.upper(), 
                   save_path=str(METRICS_DIR / f"{method}_metrics.json"))
    
    # Plot (Save only to avoid blocking loops in 'all' mode)
    # Use metrics['Accuracy'] instead of best_fit to ensure Pareto plot Y-axis is correct (Accuracy)
    plot_pareto_front(optimizer.history, len(best_names), metrics['Accuracy'], 
                      title=f"{method.upper()} Optimization History",
                      save_path=str(PLOTS_DIR / f"{method}_history.png"),
                      show=show_plots)

def run_baseline(config, df_split):
    X_train, X_test, y_train, y_test, feature_names = df_split
    
    print("\n--- Running Baseline (All Features) ---")
    
    clf = DecisionTreeClassifier(
        criterion=config['model']['decision_tree']['criterion'], 
        random_state=config['model']['decision_tree']['random_state']
    )
    
    import time
    start_time = time.time()
    clf.fit(X_train, y_train)
    runtime = time.time() - start_time
    
    print(f"Baseline Training Complete. Time: {runtime:.2f}s")
    
    evaluate_model(clf, X_test, y_test, X_train.shape[1], runtime, "BASELINE", 
                   save_path=str(METRICS_DIR / "baseline_metrics.json"))

def main():
    parser = argparse.ArgumentParser(description="IDS Feature Selection Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'compare', 'all'], help='Action to perform')
    parser.add_argument('--method', type=str, choices=['baseline', 'ga', 'pso', 'abc', 'hybrid', 'joint', 'all'], help='Optimization method to use (required for mode=train)')
    
    args = parser.parse_args()
    config = load_config()
    
    loader = IDSDataLoader(config)
    
    if args.mode == 'preprocess':
        loader.load_and_preprocess()
        
    elif args.mode == 'train':
        if not args.method:
            print("Please specify --method for training mode.")
            return
            
        data_split = loader.get_data_split()
        
        if args.method == 'all':
            run_baseline(config, data_split)
            for m in ['ga', 'pso', 'abc', 'hybrid', 'joint']:
                # Run all without blocking plots
                run_feature_selection(m, config, data_split, show_plots=False)
        elif args.method == 'baseline':
            run_baseline(config, data_split)
        else:
            # Run single with blocking plot
            run_feature_selection(args.method, config, data_split, show_plots=True)

    elif args.mode == 'compare':
        run_analysis(show_plots=True)
            
    elif args.mode == 'all':
        loader.load_and_preprocess()
        data_split = loader.get_data_split()
        
        run_baseline(config, data_split)
        for m in ['ga', 'pso', 'abc', 'hybrid', 'joint']:
            # Run all without blocking plots
            run_feature_selection(m, config, data_split, show_plots=False)
            
        print("\nAll training complete. Running final analysis...")
        run_analysis(show_plots=False)

if __name__ == "__main__":
    main()
