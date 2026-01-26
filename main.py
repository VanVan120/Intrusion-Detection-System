import argparse
import yaml
from src.data_loader import IDSDataLoader
from src.models import GASelector, PSOSelector, ABCSelector, HybridSelector, JointSelector
from src.utils import evaluate_model, plot_pareto_front
from src.analysis import run_analysis
from sklearn.tree import DecisionTreeClassifier

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_feature_selection(method, config, df_split, show_plots=True):
    X_train, X_test, y_train, y_test, feature_names = df_split
    
    optimizers = {
        'ga': GASelector,
        'pso': PSOSelector,
        'abc': ABCSelector,
        'hybrid': HybridSelector,
        'joint': JointSelector
    }
    
    if method not in optimizers:
        raise ValueError(f"Method {method} not supported. Choose from {list(optimizers.keys())}")
        
    print(f"\n--- Running {method.upper()} Optimization ---")
    optimizer = optimizers[method](X_train, y_train, feature_names, config)
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
    evaluate_model(clf, X_test_sel, y_test, len(best_indices), runtime, method.upper(), 
                   save_path=f"results/{method}_metrics.json")
    
    # Plot (Save only to avoid blocking loops in 'all' mode)
    plot_pareto_front(optimizer.history, len(best_names), best_fit, 
                      title=f"{method.upper()} Optimization History",
                      save_path=f"results/plots/{method}_history.png",
                      show=show_plots)

def main():
    parser = argparse.ArgumentParser(description="IDS Feature Selection Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'compare', 'all'], help='Action to perform')
    parser.add_argument('--method', type=str, choices=['ga', 'pso', 'abc', 'hybrid', 'joint', 'all'], help='Optimization method to use (required for mode=train)')
    
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
            for m in ['ga', 'pso', 'abc', 'hybrid', 'joint']:
                # Run all without blocking plots
                run_feature_selection(m, config, data_split, show_plots=False)
        else:
            # Run single with blocking plot
            run_feature_selection(args.method, config, data_split, show_plots=True)

    elif args.mode == 'compare':
        run_analysis()
            
    elif args.mode == 'all':
        loader.load_and_preprocess()
        data_split = loader.get_data_split()
        for m in ['ga', 'pso', 'abc', 'hybrid', 'joint']:
            # Run all without blocking plots
            run_feature_selection(m, config, data_split, show_plots=False)

if __name__ == "__main__":
    main()
