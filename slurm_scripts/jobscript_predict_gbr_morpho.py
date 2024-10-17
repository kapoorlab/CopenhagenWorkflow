import subprocess

# Define arrays for the different parameter values for model inference
model_dirs_base = [
    "morpho_feature_attention_shallowest_litest",
    "morpho_feature_attention_shallowest_liter",
    "morpho_feature_attention_shallowest_lite",
    "morpho_feature_attention_shallowest",
    "shape_feature_attention_shallowest_litest",
    "shape_feature_attention_shallowest_liter",
    "shape_feature_attention_shallowest_lite",
    "shape_feature_attention_shallowest"
]

channels = [
    "nuclei_",
    "membrane_"
]

# Loop through each model configuration and submit a separate job for each combination
for model_name in model_dirs_base:
    for channel in channels:
        
        # Construct the sbatch command to run the Python script with specified arguments
        sbatch_command = [
            "sbatch",
            "--nodes=1",
            "-A", "jsy@v100",
            "--gres=gpu:1",
             "-C", "v100-32g",
            "--cpus-per-task=40",
            "--time=20:00:00",
            f"--job-name={model_name}_{channel}",
            f"--output={model_name}_{channel}.o%j",
            f"--error={model_name}_{channel}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && python 09_inception_prediction_attention_morpho.py --dataset_name Sixth --home_folder /lustre/fsn1/projects/rech/jsy/uzj81mi/ --channel {channel} --t_initials 50 --t_finals 400 --tracklet_length 25 --model_dir /lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/ --model_name {model_name}"
        ]
        
        # Submit the job
        subprocess.run(sbatch_command)
