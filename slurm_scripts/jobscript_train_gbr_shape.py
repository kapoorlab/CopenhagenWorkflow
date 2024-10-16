import subprocess

# Define arrays for the different parameter values for shape models
shape_model_dirs_base = [
    "shape_feature_attention_shallowest_litest",
    "shape_feature_attention_shallowest_liter",
    "shape_feature_attention_shallowest_lite",
    "shape_feature_attention_shallowest"
]

block_configs = [
    "6",
    "6",
    "6",
    "6"
]

growth_rates = [4, 8, 16, 32]

channels = [
    "nuclei_",
    "membrane_"
]

shape_gbr_h5_files = [
    "shape_training_data_gbr_25_nuclei_.h5",
    "shape_training_data_gbr_25_membrane_.h5"
]

# Loop through each model configuration
for i, shape_model_base in enumerate(shape_model_dirs_base):
    block_config = block_configs[i]
    growth_rate = growth_rates[i]

    # Loop through each channel and submit a separate job
    for j, channel in enumerate(channels):
        shape_gbr_h5_file = shape_gbr_h5_files[j]
        
        # Include the channel name in the model directory
        shape_model_dir = f"/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/{shape_model_base}_{channel}/"
        
        # Construct the sbatch command
        sbatch_command = [
            "sbatch",
            "--nodes=1",
            "-A", "jsy@v100",
            "--gres=gpu:1",
            "--partition=gpu_p2",
            "--cpus-per-task=40",
            "--time=20:00:00",
            f"--job-name=Shape_{i}_{j}",
            f"--output=shape_{i}_{j}.o%j",
            f"--error=shape_{i}_{j}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/lib/ && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/ && module load cuda/11.8.0 && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/train_gbr_neural_net_shape.py --shape_model_dir {shape_model_dir} --block_config {block_config} --growth_rate {growth_rate} --channel {channel} --shape_gbr_h5_file {shape_gbr_h5_file}"
        ]
        
        # Submit the job
        subprocess.run(sbatch_command)