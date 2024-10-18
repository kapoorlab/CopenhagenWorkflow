import subprocess

# Define arrays for the different parameter values for shape models
shape_model_dirs_base = [
    "morpho_feature_attention_shallowest_litest",
    "morpho_feature_attention_shallowest_liter",
    "morpho_feature_attention_shallowest_lite",
    "morpho_feature_attention_shallowest",
]

channels = [
    "nuclei_",
    "membrane_"
]

# Loop through each model configuration
for i, shape_model_base in enumerate(shape_model_dirs_base):
    # Loop through each channel and submit a separate job
    for j, channel in enumerate(channels):
        # Construct the sbatch command
        sbatch_command = [
            "sbatch",
            "--nodes=1",
            "-A", "jsy@v100",
            #"--gres=gpu:1",
            "--partition=prepost",
            "--cpus-per-task=40",
            "--time=10:00:00",
            f"--job-name=Shape_{i}_{j}",
            f"--output=shape_{i}_{j}.o%j",
            f"--error=shape_{i}_{j}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/08_cell_type_dataframe_generator.py --dataset_name Second --home_folder /lustre/fsn1/projects/rech/jsy/uzj81mi/ --channel {channel} --model_name {shape_model_base}"
        ]
        
        # Submit the job
        subprocess.run(sbatch_command)
