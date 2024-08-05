from caped_ai_metrics import SegmentationScore
from pathlib import Path
import os

current_dir = os.path.dirname(__file__)
prediction_directory = os.path.join(current_dir, 'predictions_nuclei_vollseg')
ground_truth_directory = os.path.join(current_dir, 'ground_truth_nuclei')
result_dir = prediction_directory + '/results'
Path(result_dir).mkdir(parents=True, exist_ok=True)

compute_stats = SegmentationScore(ground_truth_directory, prediction_directory, result_dir)
dataframe = compute_stats.seg_stats()
print(dataframe)