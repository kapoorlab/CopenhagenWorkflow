import subprocess

# Define arrays for the different parameter values for morpho models
gbr_vision_model_dirs = [
    "gbr_vision"
]

channels = [
    "nuclei_",
]

batch_size = 64

vision_gbr_h5_files = [
    "cellfate_vision_training_data_gbr.h5",
]

for i, morpho_model_base in enumerate(gbr_vision_model_dirs):
    input_shape = [10,128,128,8]

    for j, channel in enumerate(channels):
        vision_gbr_h5_file = vision_gbr_h5_files[j]
        
        morpho_model_dir = f"/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/{morpho_model_base}_{channel}/"
        
        sbatch_command = [
           "sbatch",
            "--nodes=1",
            #"-A", "jsy@a100",
            #"-C", "a100",
            "-A", "jsy@v100",
            "--gres=gpu:1",
            #"--partition=gpu_p5",
            "--partition=gpu_p2",
            "--cpus-per-task=40",
            "--time=20:00:00",
            f"--job-name=Morpho_{i}_{j}",
            f"--output=morpho_{i}_{j}.o%j",
            f"--error=morpho_{i}_{j}.e%j",
            "--wrap",
            f"module purge && module load anaconda-py3 && conda deactivate && conda activate capedenv && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/lib/ && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/linkhome/rech/gengoq01/uzj81mi/.conda/envs/capedenv/ && module load cuda/11.8.0 && python /gpfswork/rech/jsy/uzj81mi/CopenhagenWorkflow/train_gbr_neural_net_vision.py --vision_model_dir {morpho_model_dir} --input_shape {input_shape} --batch_size {batch_size} --channel {channel} --vision_gbr_h5_file {vision_gbr_h5_file}"
        ]
        
        # Submit the job
        subprocess.run(sbatch_command)
        