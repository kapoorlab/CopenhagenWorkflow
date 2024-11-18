import subprocess


block_configs = [
    "6",
    "6",
    "6",
    "6",
    
   
]

growth_rates = [4, 8, 16, 32]

channels = [
    "nuclei_",
   
]

morphodynamic_mitosis_h5_files = [
    "morphodynamic_training_data_mitosis_nuclei_25.h5",
   
]

for i, growth_rate in enumerate(growth_rates):
    block_config = block_configs[i]

    for j, channel in enumerate(channels):
        morphodynamic_mitosis_h5_file = morphodynamic_mitosis_h5_files[j]
        
        morpho_model_dir = f"/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/morphodynamic_feature_mitosis_25_growth_rate_{growth_rate}/"
        
        sbatch_command = [
            "sbatch",
            "--nodes=1",
            "-A", "jsy@a100",
            "-C", "a100",
            #"-A", "jsy@v100",
            "--gres=gpu:1",
            "--partition=gpu_p5",
            #"--partition=gpu_p2",
            "--cpus-per-task=40",
            "--time=20:00:00",
            f"--job-name=Morpho_{i}_{j}",
            f"--output=morpho_{i}_{j}.o%j",
            f"--error=morpho_{i}_{j}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/lib/ && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/ && module load cuda/11.8.0 && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/train_mitosis_neural_net_morphodynamic_argparse.py --morpho_model_dir {morpho_model_dir} --block_config {block_config} --growth_rate {growth_rate}  --morphodynamic_mitosis_h5_file {morphodynamic_mitosis_h5_file}"
        ]
        
        # Submit the job
        subprocess.run(sbatch_command)
        