#!/usr/bin/env python3
"""
Top-level comparison of two TrackMate XMLs without CLI parsing,
computing assignment-only metrics plus CCA and CT, and summary plots.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from napatrackmater.Trackcomparator import TrackComparator

# --- Config: set your file paths and matching threshold here ---
output_dir  = (
    'tracking_accuracy_results/'
)
gt_tracking_directory = (
    '/nuclei_tracking_gt/'
)
pred_tracking_directory = (
    '/nuclei_tracking_pred/'
)
xml_name        = 'hyperstack.xml'
MATCH_THRESHOLD = 10.0   # µm
DOWNSAMPLE      = 1      # time downsampling factor

# ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

gt_xml   = os.path.join(gt_tracking_directory, xml_name)
pred_xml = os.path.join(pred_tracking_directory, xml_name)

# --- Compare tracks and compute metrics ---
tc      = TrackComparator(gt=gt_xml, pred=pred_xml, downsampleT=DOWNSAMPLE)
results = tc.evaluate(threshold=MATCH_THRESHOLD)
assignments = results['assignments']
num_hits    = results['num_hits']
num_gt      = results['num_gt']
num_pred    = results['num_pred']
cca         = results['cca']
ct          = results['ct']

# --- Save assignments to CSV ---
asg_file = os.path.join(output_dir, 'track_assignments.csv')
assignments.to_csv(asg_file, index=False)

# --- Print summary metrics ---
print(f"GT tracks:            {num_gt}")
print(f"Predicted tracks:     {num_pred}")
print(f"Threshold:            {MATCH_THRESHOLD} µm")
print(f"Correct matches:      {num_hits}")
print(f"Fraction matched:     {num_hits/num_gt:.2%}")
print(f"Cell Cycle Accuracy:  {cca:.3f}")
print(f"Complete Tracks CT:   {ct:.3f}")
print(f"Saved assignments →   {asg_file}")

# --- Summary Plots of assignment distances ---
# 1. Histogram of assignment distances
plt.figure(figsize=(6,4))
plt.hist(assignments['distance'], bins=30, edgecolor='black')
plt.axvline(MATCH_THRESHOLD, color='red', linestyle='--',
            label=f'threshold = {MATCH_THRESHOLD} µm')
plt.xlabel('Assigned Track Distance (µm)')
plt.ylabel('Count')
plt.title('Histogram of Assigned Track Distances')
plt.legend()
hist_file = os.path.join(output_dir, 'assignment_distance_histogram.png')
plt.savefig(hist_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved histogram → {hist_file}")

# 2. Boxplot of assignment distances
plt.figure(figsize=(4,6))
plt.boxplot(assignments['distance'], vert=True)
plt.ylabel('Distance (µm)')
plt.title('Boxplot of Assigned Track Distances')
box_file = os.path.join(output_dir, 'assignment_distance_boxplot.png')
plt.savefig(box_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved boxplot → {box_file}")

# 3. Cumulative distribution of assignment distances
sorted_dist = assignments['distance'].sort_values().reset_index(drop=True)
cumvals = (sorted_dist.index + 1) / len(sorted_dist)
plt.figure(figsize=(6,4))
plt.plot(sorted_dist, cumvals, marker='.', linestyle='none')
plt.axvline(MATCH_THRESHOLD, color='red', linestyle='--',
            label=f'threshold = {MATCH_THRESHOLD} µm')
plt.xlabel('Assigned Track Distance (µm)')
plt.ylabel('Cumulative Fraction')
plt.title('CDF of Assigned Track Distances')
plt.legend()
cdf_file = os.path.join(output_dir, 'assignment_distance_cdf.png')
plt.savefig(cdf_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved CDF plot → {cdf_file}")

# --- Metrics Barplot for CCA and CT ---
metrics = [cca, ct]
labels = ['CCA', 'CT']
plt.figure(figsize=(4,4))
plt.bar(labels, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Track-level Metrics')
metrics_file = os.path.join(output_dir, 'cca_ct_barplot.png')
plt.savefig(metrics_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved metrics barplot → {metrics_file}")