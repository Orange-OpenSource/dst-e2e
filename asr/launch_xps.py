# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

#!/usr/bin/python3

import os

asr_models = [
    #"wav2vec",
    "wavlm"
]

token_types = [
    # "char",
    # "unigram",
    "bpe"
]

number_tokens = [
    # 500,
    # 1000,
    5000,
    # 10000
]

lrs = [
    0.0001, 
    #0.00001
    ]
factors = [
    #1000,
    10000,
    #100000
]
virtual_batches = [64]
epochs = 10
nbr_examples_train = 57000

for asr_model in asr_models:
    for lr in lrs:
        for factor in factors:
            for virtual_batch in virtual_batches:
                for token_type in token_types:
                    if token_type in ["unigram", "bpe"]:
                        for number_token in number_tokens:

                            os.system(f"sbatch --constraint=gpu_mem_24 --job-name=ASR_{asr_model}_{token_type}_{number_token}_{lr:f}_{factor}_{virtual_batch}" +
                            " run.sbatch python3.8 train_asr.py hparams/train_asr.yaml" +
                            f" --number_of_epochs {epochs}" +
                            f' --wav2vec2_hub {"facebook/wav2vec2-base-960h" if asr_model == "wav2vec" else "microsoft/wavlm-base-plus"}' +
                            f" --lr {lr*factor:f} --lr_wav2vec {lr:f}" +
                            f' --batch_size 4' + 
                            f' --grad_accumulation_factor {virtual_batch//4}' +
                            f' --warmup_steps {epochs * nbr_examples_train // (10*virtual_batch)}' +
                            f' --token_type {token_type}' +
                            f' --output_neurons {number_token}' +
                            f' --inference True' +
                            f' --output_folder /results/ASR_{asr_model}_{token_type}_{number_token}_{lr:f}_{factor}_{virtual_batch}'
                            )

                    else:
                        
                        os.system(f"sbatch --constraint=gpu_mem_24 --job-name=ASR_{asr_model}_{token_type}_{lr:f}_{factor}_{virtual_batch}" +
                        " run.sbatch python3.8 train_asr.py hparams/train_asr.yaml" +
                        f" --number_of_epochs {epochs}" +
                        f' --wav2vec2_hub {"facebook/wav2vec2-base-960h" if asr_model == "wav2vec" else "microsoft/wavlm-base-plus"}' +
                        f" --lr {lr*factor:f} --lr_wav2vec {lr:f}" +
                        f' --batch_size 4' + 
                        f' --grad_accumulation_factor {virtual_batch//4}' +
                        f' --warmup_steps {epochs * nbr_examples_train // (10*virtual_batch)}' +
                        f' --token_type {token_type}' +
                        f' --output_neurons 60' +
                        f' --output_folder /results/ASR_{asr_model}_{token_type}_{lr:f}_{factor}_{virtual_batch}'
                        )


