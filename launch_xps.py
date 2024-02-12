#!/usr/bin/python3
import os

datasets = [
    "multiwoz",
    # "spokenwoz"
    ]

number_examples = {
    "multiwoz": 56750,
    "spokenwoz": 74524
}

versions = [
    "cascade-gt",
    # "cascade-nlu", 
    # "cascade-whisper", 
    # "global",
    # "global-whisper"
    ]

folders = [
    "w2v+t5",
    # "whisperSLU"
    ]

num_cpu = 8

lrs = [
    # 0.001,
    0.0001,
    # 0.00001,
    # 0.000001 
]

inference = True
gold_previous_state = False

commands = []
# Dialogue State Tracking
for folder in folders:
    for dataset in datasets:
        for lr in lrs:

            if folder == "w2v+t5":
                for version in versions:

                    # Re-init params for each version
                    # FIXME: Issue with cuda version of 4090 cards --> new dockerfile with cu11.8 --> remove enroot cache
                    # "gpu_3090|gpu_A100"
                    sbatch_params = f' --constraint="gpu_3090|gpu_A100" --time 8:00:00 --cpus-per-task={num_cpu}'
                    yaml_params = f' --lr {lr:f}'
                    
                    if dataset == "multiwoz":
                        yaml_params += f' --data_folder /data/AUDIO/corpus/dstc11'
                    else:
                        yaml_params += f' --data_folder /data/AUDIO/corpus/SpokenWoz'

                    sbatch_params +=  f" --job-name=dstE2E-{dataset}_{version}_{lr:f}"
                    
                    if version == "global-whisper":
                        yaml_file = f"{folder}/hparams/train_{dataset}_with_whisper_enc.yaml"
                        yaml_params += ' --version global'
                        yaml_params += " --freeze_feature_extractor False --reinitialize_last_layers False"
                    else:
                        yaml_file = f"{folder}/hparams/train_{dataset}.yaml"
                        yaml_params += f' --version {version}'

                    if "global" in version:
                        yaml_params += ' --number_of_epochs 20'
                        # For the same scheduler behavior for the 10 first epochs
                        warmup_steps = 20*number_examples[dataset] // (10*4*16)
                        yaml_params += f' --warmup_steps {warmup_steps}'
                        yaml_params += f' --n_epochs_valid 4'
                    
                    yaml_params += f' --output_folder /results/dstE2E_{dataset}_{version}_{lr:f}'
                    # Leave default workers when inference with gold_previous_state False
                    # yaml_params += f' --num_workers {num_cpu}'

                    if inference:
                        yaml_params += ' --inference True'
                        yaml_params += f' --gold_previous_state {gold_previous_state}'

                    commands.append(f"sbatch{sbatch_params} run.sbatch python {folder}/train_{dataset}.py {yaml_file}{yaml_params}")

            else:
                version = "local"
                sbatch_params = f' --constraint="gpu_3090|gpu_A100" --time 14:00:00 --cpus-per-task={num_cpu} --job-name=dstE2E-{dataset}_local_{lr:f}'
                
                yaml_params = f' --lr {lr:f}'
                
                if dataset == "multiwoz":
                    yaml_params += f' --data_folder /data/AUDIO/corpus/dstc11'
                else:
                    yaml_params += f' --data_folder /data/AUDIO/corpus/SpokenWoz'
                
                yaml_file = f"{folder}/hparams/train_{dataset}.yaml"
                
                yaml_params += f' --output_folder /results/dstE2E_{dataset}_{version}_{lr:f}'
                yaml_params += f' --num_workers {num_cpu}'
                yaml_params += f' --version local'
                
                if inference:
                    yaml_params += ' --inference True'
                    yaml_params += f' --gold_previous_state {gold_previous_state}'
                
                commands.append(f"sbatch{sbatch_params} run.sbatch python {folder}/train_{dataset}.py {yaml_file}{yaml_params}")

# Transcriptions with whisper
# for dataset in datasets:
#     base_command = f" run.sbatch python asr/whisper_transcribe.py --dataset {dataset}"
#     base_command += " --output_folder /results/whisper_transcriptions"

#     if dataset == "multiwoz":
#         base_command += " --data_folder /data/AUDIO/corpus/dstc11"
#     else:
#         base_command += " --data_folder /data/AUDIO/corpus/SpokenWoz"
    
#     command = f'sbatch --constraint=gpu_mem_24 --time 12:00:00 --job-name=whisper_transcribe_{dataset}' + base_command
#     commands.append(command)
    
for command in commands:
    os.system(command)