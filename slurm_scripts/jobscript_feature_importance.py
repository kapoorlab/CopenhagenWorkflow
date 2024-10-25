import subprocess

# Define parameters for model inference
model_name = "morpho_feature_attention_shallowest_litest"
channel = "nuclei_"
dataset = "Sixth"  

# Construct the sbatch command to run the Python script with specified arguments
sbatch_command = [
    "sbatch",
    "--nodes=1",
    "-A", "jsy@v100",
    "--partition=prepost",
    "--cpus-per-task=40",
    "--time=10:00:00",
    f"--job-name={model_name}_{channel}_{dataset}",
    f"--output={model_name}_{channel}_{dataset}.o%j",
    f"--error={model_name}_{channel}_{dataset}.e%j",
    "--wrap",
    f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/09_inception_prediction_attention_morpho.py --dataset_name {dataset} --home_folder /lustre/fsn1/projects/rech/jsy/uzj81mi/ --channel {channel} --t_initials 50 --t_finals 400 --tracklet_length 25 --model_dir /lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/ --model_name {model_name}"
]

# Submit the job
subprocess.run(sbatch_command)
