# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

#!/usr/bin/env python

import sys
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

from pathlib import Path
import os
import pandas as pd

from whisper_prompt_context import WhisperSLU

def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_datasets = {}
    for csv_file in hparams["valid_csv"]:
        name = Path(csv_file).stem
        valid_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        valid_datasets[name] = valid_datasets[name].filtered_sorted(
            sort_key="duration"
        )
    valid_data = valid_datasets[Path(hparams["valid_csv"][0]).stem]

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem

        if hparams["sorting_turns"]:
            # Sorting by turn nbr to always have the previous dialogue state already processed, 
            # default is ascending
            ordered_csv = csv_file.replace(".csv", "_sorted.csv")
            df = pd.read_csv(csv_file)
            df.sort_values(by="turnID", inplace=True)
            df.to_csv(ordered_csv, header=True, index=False)
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=ordered_csv, replacements={"data_root": data_folder}
            )
        else:
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration"
            )
        hparams["valid_loader_kwargs"]["shuffle"] = False

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("id", "agent", "previous_state", "current_state", "transcription")
    @sb.utils.data_pipeline.provides(
        "agent", "semantics", "semantics_tokens", "outputs", "outputs_tokens", "outputs_tokens_nobos", "transcription"
    )
    def text_pipeline(ID, agent, previous_state, current_state, transcription):

        yield agent

        if hparams["gold_previous_state"]:
            semantics = previous_state
        else:
            dialogue_id = ID.split("/")[-2]
            turn_id = int(ID.split("/")[-1].split("-")[-1])
            if turn_id > 1:
                dialogue_last_turn = ""
                assert(os.path.isfile(os.path.join(hparams["output_folder"], "last_turns", f'{dialogue_id}.txt')))
                with open(os.path.join(hparams["output_folder"], "last_turns", f'{dialogue_id}.txt'), "r") as last_turn:
                    for line in last_turn:
                        dialogue_last_turn = line.strip()
                semantics = dialogue_last_turn
            else:
                semantics = ""
        yield semantics

        semantics_tokens = tokenizer.encode(f"[State] {semantics} [Agent] {agent}")
        
        semantics_tokens = torch.LongTensor(semantics_tokens)
        yield semantics_tokens

        outputs = current_state
        yield outputs

        # Whisper tokenizer adds automatically sot, lang, task and timestamp tokens at the beginning and eot at the end
        if hparams["output_mode"] == "transcription":
            tokens_list = tokenizer.encode(f'[User] {transcription} [State] {current_state}')
        elif hparams["output_mode"] == "state":
            tokens_list = tokenizer.encode(f'[State] {current_state}')
        else:
            raise KeyError('hparams attribute "output_mode" should be set to "transcription" or "state".')
        
        yield torch.LongTensor(tokens_list[:-1])

        yield torch.LongTensor(tokens_list[1:])

        yield transcription

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "agent", "semantics_tokens", "sig", "outputs_tokens", "outputs_tokens_nobos"],
    )

    return train_data, valid_data, test_datasets

if __name__=="__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams["skip_prep"]:

        # Dataset preparation
        from utils.multiwoz_prepare_slu import prepare_multiwoz

        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_multiwoz,
            kwargs={
                "data_folder": hparams["data_folder"],
                "version": hparams["version"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["save_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
                "select_n_sentences": hparams["select_n_sentences"],
            },
        )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer
    tokenizer.set_prefix_tokens(hparams["language"], "transcribe", False)

    if hparams["gold_previous_state"]:
        hparams["sorting_turns"] = False
    else:
        hparams["sorting_turns"] = True

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    slu_brain = WhisperSLU(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )
    slu_brain.tokenizer = tokenizer

    if not hparams["inference"]:
        slu_brain.fit(
            slu_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"],
            n_epochs_valid=2
        )

    else:
        # Testing
        for k in test_datasets.keys():
            slu_brain.hparams.output_file = os.path.join(
                hparams["pred_folder"], "{}.csv".format(k)
            )
            if not hparams["gold_previous_state"]:
                # Storing the last dialog's turn prediction
                if not os.path.isdir(os.path.join(hparams["output_folder"], "last_turns")):
                    os.mkdir(os.path.join(hparams["output_folder"], "last_turns"))
                slu_brain.hparams.output_file = os.path.join(
                    hparams["pred_folder"], "{}_previous.csv".format(k)
                )
                hparams["valid_loader_kwargs"]["num_workers"] = 1
            if not os.path.isfile(slu_brain.hparams.output_file):
                slu_brain.evaluate(
                    test_datasets[k], test_loader_kwargs=hparams["valid_loader_kwargs"]
                    )
                # Requeuing after finished inference
                sys.exit(42)