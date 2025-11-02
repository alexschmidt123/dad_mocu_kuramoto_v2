"""
Evaluate MOCU Predictors

Compare different MOCU prediction models on:
1. Prediction accuracy (MSE, MAE)
2. Time complexity (inference speed)
3. Model size

Usage:
    python scripts/evaluate_predictors.py --test_data ./data/test_data.pt
"""

import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predictors import MLPPredictor, CNNPredictor, MPNNPlusPredictor
from src.models.predictors.predictors import SamplingBasedMOCU  # Direct import to avoid __init__ side effects
from src.core.mocu_cuda import MOCU


def load_test_data(test_data_path):
    """Load test dataset."""
    print(f"Loading test data from {test_data_path}...")
    test_data = torch.load(test_data_path)
    return test_data


def evaluate_predictor(predictor, test_loader, device, predictor_name):
    """
    Evaluate a single predictor.
    
    Returns:
        metrics: Dict with MSE, MAE, time, etc.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {predictor_name}")
    print(f"{'='*80}")
    
    predictor.eval()
    predictor.to(device)
    
    predictions = []
    targets = []
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Measure inference time
            start_time = time.time()
            pred = predictor(batch)
            inference_time = time.time() - start_time
            
            predictions.append(pred.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
            inference_times.append(inference_time)
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Relative errors
    relative_errors = np.abs((predictions - targets) / (targets + 1e-10))
    mean_relative_error = np.mean(relative_errors)
    
    # Time complexity
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    # Model size
    param_count = sum(p.numel() for p in predictor.parameters())
    model_size_mb = param_count * 4 / (1024 ** 2)  # Assuming float32
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Mean Relative Error': mean_relative_error,
        'Avg Inference Time (s)': avg_inference_time,
        'Total Inference Time (s)': total_inference_time,
        'Param Count': param_count,
        'Model Size (MB)': model_size_mb,
    }
    
    # Print metrics
    print(f"\n📊 Prediction Accuracy:")
    print(f"  MSE:                  {mse:.6f}")
    print(f"  MAE:                  {mae:.6f}")
    print(f"  RMSE:                 {rmse:.6f}")
    print(f"  Mean Relative Error:  {mean_relative_error:.4%}")
    
    print(f"\n⏱️  Time Complexity:")
    print(f"  Avg Inference Time:   {avg_inference_time:.6f} s")
    print(f"  Total Time:           {total_inference_time:.4f} s")
    print(f"  Throughput:           {len(predictions) / total_inference_time:.2f} samples/s")
    
    print(f"\n💾 Model Size:")
    print(f"  Parameters:           {param_count:,}")
    print(f"  Size:                 {model_size_mb:.2f} MB")
    
    return metrics, predictions, targets


def evaluate_sampling_based(test_data, N, K_max, deltaT, MReal, TReal):
    """Evaluate sampling-based (ground truth) method."""
    print(f"\n{'='*80}")
    print(f"Evaluating Sampling-Based (Ground Truth)")
    print(f"{'='*80}")
    
    sampler = SamplingBasedMOCU(K_max=K_max, h=deltaT, T=TReal)
    
    predictions = []
    targets = []
    inference_times = []
    
    for data in test_data:
        # Extract w, a_lower, a_upper from data
        w = data.x.numpy().flatten()
        # Need to reconstruct a_lower and a_upper from edge_attr
        # This depends on how data is structured
        
        start_time = time.time()
        # pred = sampler.compute(w, a_lower, a_upper)
        # For now, skip sampling-based evaluation as it requires specific data format
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    print("⚠️  Sampling-based evaluation requires specific data format")
    print("   Skipping for now - use for reference only")
    
    return None, None, None


def compare_predictors(metrics_dict):
    """Generate comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Accuracy comparison
    print("📊 Prediction Accuracy (Lower is Better):")
    print(f"{'Method':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    for name, metrics in metrics_dict.items():
        print(f"{name:<20} {metrics['MSE']:<12.6f} {metrics['MAE']:<12.6f} {metrics['RMSE']:<12.6f}")
    
    # Speed comparison
    print(f"\n⏱️  Inference Speed (Lower is Better):")
    print(f"{'Method':<20} {'Avg Time (s)':<15} {'Throughput (samples/s)':<25}")
    print("-" * 60)
    for name, metrics in metrics_dict.items():
        throughput = 1.0 / metrics['Avg Inference Time (s)'] if metrics['Avg Inference Time (s)'] > 0 else 0
        print(f"{name:<20} {metrics['Avg Inference Time (s)']:<15.6f} {throughput:<25.2f}")
    
    # Model size comparison
    print(f"\n💾 Model Size:")
    print(f"{'Method':<20} {'Parameters':<15} {'Size (MB)':<12}")
    print("-" * 60)
    for name, metrics in metrics_dict.items():
        print(f"{name:<20} {metrics['Param Count']:<15,} {metrics['Model Size (MB)']:<12.2f}")
    
    # Best methods
    print(f"\n🏆 Best Methods:")
    best_mse = min(metrics_dict.items(), key=lambda x: x[1]['MSE'])
    best_speed = min(metrics_dict.items(), key=lambda x: x[1]['Avg Inference Time (s)'])
    smallest = min(metrics_dict.items(), key=lambda x: x[1]['Param Count'])
    
    print(f"  Best Accuracy:  {best_mse[0]} (MSE: {best_mse[1]['MSE']:.6f})")
    print(f"  Fastest:        {best_speed[0]} (Time: {best_speed[1]['Avg Inference Time (s)']:.6f} s)")
    print(f"  Smallest:       {smallest[0]} (Params: {smallest[1]['Param Count']:,})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MOCU Predictors')
    parser.add_argument('--test_data', type=str, default='./data/test_data.pt',
                        help='Path to test dataset')
    parser.add_argument('--N', type=int, default=5,
                        help='Number of oscillators')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory containing trained models')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_data = load_test_data(args.test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Dictionary to store metrics
    metrics_dict = {}
    
    # Evaluate MLP
    print("\n" + "="*80)
    print("EVALUATING MLP PREDICTOR")
    print("="*80)
    mlp_model = MLPPredictor(args.N).to(device)
    mlp_path = Path(args.models_dir) / 'mlp_predictor' / 'model.pth'
    if mlp_path.exists():
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
        metrics_dict['MLP'], _, _ = evaluate_predictor(mlp_model, test_loader, device, 'MLP')
    else:
        print(f"⚠️  MLP model not found at {mlp_path}")
    
    # Evaluate CNN
    print("\n" + "="*80)
    print("EVALUATING CNN PREDICTOR")
    print("="*80)
    cnn_model = CNNPredictor(args.N).to(device)
    cnn_path = Path(args.models_dir) / 'cnn_predictor' / 'model.pth'
    if cnn_path.exists():
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        metrics_dict['CNN'], _, _ = evaluate_predictor(cnn_model, test_loader, device, 'CNN')
    else:
        print(f"⚠️  CNN model not found at {cnn_path}")
    
    # Evaluate MPNN+
    print("\n" + "="*80)
    print("EVALUATING MPNN+ PREDICTOR")
    print("="*80)
    mpnn_model = MPNNPlusPredictor(args.N).to(device)
    mpnn_path = Path(args.models_dir) / 'cons5' / 'model.pth'
    if mpnn_path.exists():
        mpnn_model.load_state_dict(torch.load(mpnn_path, map_location=device))
        metrics_dict['MPNN+'], _, _ = evaluate_predictor(mpnn_model, test_loader, device, 'MPNN+')
    else:
        print(f"⚠️  MPNN+ model not found at {mpnn_path}")
    
    # Compare all predictors
    if metrics_dict:
        compare_predictors(metrics_dict)
    
    # Save results
    output_file = PROJECT_ROOT / 'results' / 'predictor_evaluation.txt'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("MOCU Predictor Evaluation Results\n")
        f.write("="*80 + "\n\n")
        for name, metrics in metrics_dict.items():
            f.write(f"{name}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()

