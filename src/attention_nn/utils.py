from pathlib import Path

def get_weights_file_path(config: dict, epoch: str) -> str:
    """ Get the file path for saving/loading model weights for a specific epoch.
     Args:
        config (dict): Configuration dictionary containing model parameters.
            - model_folder (str): The folder name for storing model weights.
            - model_basename (str): The base name for the model weight files.
        epoch (str): The epoch identifier (e.g., 'latest' or a specific epoch number).
    Returns:
        str: The full file path for the model weights.
            Path format: ./logs/<model_folder>/<model_basename><epoch>.pt
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config:dict) -> str|None:
    """ Get the latest model weights file path from the model folder.
    Args:
        config (dict): Configuration dictionary containing model parameters.
            - model_folder (str): The folder name for storing model weights.
            - model_basename (str): The base name for the model weight files.
    Returns:
        str|None: The file path of the latest model weights, or None if no weights found.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])