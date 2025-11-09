"""
Centralized path configuration for the project.
All scripts should use functions from this module to get consistent paths.
"""
from pathlib import Path
import os


# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_experiment_root(experiment_id: str = None) -> Path:
    """
    Get the root directory for an experiment.
    
    Args:
        experiment_id: Experiment identifier (e.g., 'fast_config_20251102_095743').
                      If None, tries to get from MOCU_MODEL_NAME env var.
    
    Returns:
        Path to experiment root directory (experiments/{experiment_id}/)
    """
    if experiment_id is None:
        experiment_id = os.getenv('MOCU_MODEL_NAME')
    
    if experiment_id is None:
        raise ValueError(
            "experiment_id must be provided or MOCU_MODEL_NAME environment variable must be set"
        )
    
    return PROJECT_ROOT / 'experiments' / experiment_id


def get_data_dir(experiment_id: str = None) -> Path:
    """Get the data directory for an experiment."""
    return get_experiment_root(experiment_id) / 'data'


def get_model_dir(experiment_id: str = None) -> Path:
    """Get the model directory for an experiment."""
    return get_experiment_root(experiment_id) / 'models'


def get_result_dir(experiment_id: str = None) -> Path:
    """Get the result directory for an experiment."""
    # Try RESULT_FOLDER env var first (for backward compatibility)
    result_folder = os.getenv('RESULT_FOLDER')
    if result_folder:
        return Path(result_folder)
    
    # Otherwise use experiment-based path
    return get_experiment_root(experiment_id) / 'results'


def get_train_file(train_size: int, N: int, experiment_id: str = None) -> Path:
    """Get the training dataset file path."""
    return get_data_dir(experiment_id) / f'{train_size}_{N}o_train.pth'


def get_test_file(test_size: int, N: int, experiment_id: str = None) -> Path:
    """Get the test dataset file path."""
    return get_data_dir(experiment_id) / f'{test_size}_{N}o_test.pth'


def get_model_file(experiment_id: str = None) -> Path:
    """Get the model file path."""
    return get_model_dir(experiment_id) / 'model.pth'


def get_statistics_file(experiment_id: str = None) -> Path:
    """Get the statistics file path."""
    return get_model_dir(experiment_id) / 'statistics.pth'


def ensure_dirs(experiment_id: str = None):
    """Ensure all experiment directories exist."""
    get_data_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    get_model_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    get_result_dir(experiment_id).mkdir(parents=True, exist_ok=True)

