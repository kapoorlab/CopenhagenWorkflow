import subprocess

# Define parameters for shape models
shape_model_base = "morpho_feature_attention_shallowest_litest"
channel = "nuclei_"
datasets = ["Second", "Third", "Fifth"]

# Loop through each dataset and submit a separate job for each combination
for i, dataset in enumerate(datasets):
    # Construct the sbatch command
    sbatch_command = [
        "sbatch",
        "--nodes=1",
        "-A", "jsy@v100",
        "--partition=prepost",
        "--cpus-per-task=40",
        "--time=10:00:00",
        f"--job-name=Morpho_{i}",
        f"--output=shape_{i}.o%j",
        f"--error=shape_{i}.e%j",
        "--wrap",
        f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/08_cell_type_dataframe_generator.py --dataset_name {dataset} --home_folder /lustre/fsn1/projects/rech/jsy/uzj81mi/ --channel {channel} --model_name {shape_model_base}"
    ]
    
    # Submit the job
    subprocess.run(sbatch_command)
