import subprocess

gbr_vision_model_dirs = [
    "vision_track_training_data"
]


vision_gbr_h5_files = [
    "cellfate_vision_training_data_gbr.h5",
]

for i, morpho_model_base in enumerate(gbr_vision_model_dirs):
        

        vision_gbr_h5_file = vision_gbr_h5_files[i]
        
        morpho_model_dir = f"/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/vision_track_training_data/"
        
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
            f"--job-name=Morpho_{i}",
            f"--output=morpho_{i}.o%j",
            f"--error=morpho_{i}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/lib/ && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/ && module load cuda/11.8.0 && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/train_gbr_neural_net_vision.py --vision_model_dir {morpho_model_dir} --vision_gbr_h5_file {vision_gbr_h5_file}"
        ]
        
        subprocess.run(sbatch_command)
        