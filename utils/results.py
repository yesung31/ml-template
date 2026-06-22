import json
from pathlib import Path


def save_test_results(results, dm, log_dir):
    """
    Saves the raw test results to a JSON file.
    Can be customized to format results based on the DataModule.
    """
    if isinstance(results, list) and len(results) == 1:
        results = results[0]
        
    with open(Path(log_dir) / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
