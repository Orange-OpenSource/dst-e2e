# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

"""
Data preparation for the Multiwoz dataset vocalized by the DSTC11 Speech Aware challenge.

Download: 
    - https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html
Prerequisite: 
    - Extract the audio of each turn with the script ./extract_audio.py

Author
------
Lucas Druart 2023
"""

import os
import csv
import logging
import torchaudio
from tqdm.contrib import tzip
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
)

from utils.evaluatePredictions import cumulate, dialogueState_str2dict, dialogueState_dict2str

logger = logging.getLogger(__name__)
OPT_FILE = "opt_multiwoz_prepare.pkl"
SAMPLERATE = 16000

def prepare_multiwoz(
    data_folder,
    save_folder,
    version="global",
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    skip_prep=False,
):
    """
    This class prepares the csv files for the Multi-Woz Dialogue State Tracking dataset.
    Download: 
        - https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html
    Prerequisite: 
        - Extract the audio of each turn with the script ./extract_audio.py

    Arguments
    ---------
    data_folder : str
        Path to the folder where the extracted turn audio files are stored.
    version : str
        Version of dataset to prepare ("cascade" using previously computed transcriptions, 
        "local" using turn level states or "global" using dialogue level states).
    tr_splits : list
        List of train splits to prepare from ['train_tts'].
    dev_splits : list
        List of dev splits to prepare from ['dev_tts', 'dev_human'].
    te_splits : list
        List of test splits to prepare from ['test_tts', 'test_human', 'test_paraphrased'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of splits (e.g, train-human, train-tts,..) to merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.

    Example
    -------
    >>> data_folder = 'datasets/multi-Woz'
    >>> versions = 'global'
    >>> tr_splits = ['train_tts']
    >>> dev_splits = ['dev_tts']
    >>> te_splits = ['test_tts']
    >>> save_folder = 'multiwoz_prepared'
    >>> prepare_multiwoz(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder has the correct architecture
    check_multiwoz_folders(data_folder, version, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):

        split = splits[split_index]
        
        wav_lst = []
        text_dict = {}
        wav_lst.extend(get_all_files(
            os.path.join(data_folder, "DSTC11_" + split), match_and=[".wav"]
        ))

        # Annotations are independent from the vocalisation (TTS, Human)
        if version == "cascade-nlu":
            split_manifest = os.path.join(data_folder, split + "_asr_manifest.txt")
        elif version == "cascade-whisper":
            split_manifest = os.path.join(data_folder, split + "_whisper_manifest.txt")
        else:
            split_manifest = os.path.join(data_folder, split.split("_")[0] + "_manifest.txt")
        dialog_id = ""

        with open(split_manifest, "r") as annotations:
            for line in annotations:
                if line.__contains__("END_OF_DIALOG"):
                    pass
                else:
                    item = line.split(' ', 7)
                    # A line looks like: 
                    # line_nr: [N] dialog_id: [D.json] turn_id: [T] text: (user:|agent:) [ABC] state: domain1-slot1=value1; domain2-slot2=value2 
                    key_map = {"line_nr": 1, "dialog_id": 3, "turn_id": 5, "text": 7}
                    if item[key_map["dialog_id"]].split(".json")[0] != dialog_id:
                        # New dialog
                        dialog_id = item[key_map["dialog_id"]].split(".json")[0]
                        agent_transcription = ""
                    turn_id = item[key_map["turn_id"]]

                    # User turn
                    if int(turn_id) % 2 == 1:
                        wav_path = os.path.join(data_folder, "DSTC11_" + split, dialog_id, "Turn-" + turn_id + ".wav")
                        
                        # Extracting the text part (transcription and state) of the line
                        text_split = item[key_map["text"]].split("state:")
                        user_transcription = text_split[0].split("user:")[-1].strip()
                        state = text_split[-1].strip()

                        # Fetching the previous state  
                        if int(turn_id) > 2:
                            previous_turn_path = os.path.join(data_folder, "DSTC11_" + split, dialog_id, 
                                                              "Turn-" + str(int(turn_id) - 2) + ".wav")
                            previous_state = text_dict[previous_turn_path.split(".wav")[0] + "_current"]
                            if version == "local":
                                # previous state corresponds to the cumulation of the previous' state previous state
                                # and the previous' state current state
                                previous_previous_state = dialogueState_str2dict(text_dict[previous_turn_path.split(".wav")[0] + "_previous"])
                                previous_state = cumulate(dialogueState_str2dict(previous_state), previous_previous_state)
                                previous_state = dialogueState_dict2str(previous_state)
                                text_dict[wav_path.split(".wav")[0] + "_previous"] = previous_state
                            else:
                                text_dict[wav_path.split(".wav")[0] + "_previous"] = previous_state
                        else:
                            text_dict[wav_path.split(".wav")[0] + "_previous"] = ""

                        if version == "local" and int(turn_id) > 2:
                            # Selecting the slots which come from this agent+user turn
                            local_state = []
                            for slot in state.split(";"):
                                slot = slot.strip()
                                if "=" not in slot:
                                    continue
                                slot_type, slot_value = slot.split("=")[0].strip(), slot.split("=")[1].strip()
                                if slot not in previous_state:
                                    # New slots are either said by a speaker or references to previous slots
                                    if slot_value in agent_transcription or slot_value in user_transcription or slot_value in ["yes", "no", "dontcare"]:
                                        local_state.append(f'{slot_type}={slot_value}')
                                    else:
                                        # If the value is neither in the agent's turn nor in the user's turn
                                        # Then replacing the value by the name of a corresponding slot from the previous state
                                        for input_slot in previous_state.split(";"):
                                            input_slot = input_slot.strip()
                                            if "=" not in input_slot:
                                                continue
                                            input_slot_type, input_slot_value = input_slot.split("=")[0].strip(), input_slot.split("=")[1].strip()
                                            if input_slot_value == slot_value:
                                                local_state.append(f'{slot_type}={input_slot_type}')
                                                break

                            # Marking suppressed slots by assigning them the <unk> token
                            for slot in previous_state.split(";"):
                                if "=" not in slot:
                                    continue
                                slot_type, slot_value = slot.split("=")[0].strip(), slot.split("=")[1].strip()
                                if slot_type not in state:
                                    local_state.append(f'{slot_type}=<unk>')
                            local_state = "; ".join(local_state)
                            text_dict[wav_path.split(".wav")[0] + "_current"] = local_state

                        elif version == "cascade-nlu" or version == "cascade-gt" or \
                            version == "cascade-whisper" or version == "global" or \
                            (version == "local" and int(turn_id) <= 2):
                            text_dict[wav_path.split(".wav")[0] + "_current"] = state

                        else:
                            err_msg = (
                                "Asked for %s version of the pre-processing which does not exist."
                                "Please select one of local, global, cascade-(nlu|whisper|gt)" % version
                            )
                            raise OSError(err_msg)

                        text_dict[wav_path.split(".wav")[0] + "_transcription"] = user_transcription
                        text_dict[wav_path.split(".wav")[0] + "_agent"] = agent_transcription
                    
                    else:
                        # Extracting the text part (transcription) of the line
                        text_split = item[key_map["text"]].split("state:")
                        agent_transcription = text_split[0].split("agent:")[-1].strip()

        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences
        else:
            n_sentences = len(wav_lst)

        create_csv(
            save_folder, wav_lst, text_dict, split, n_sentences,
        )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split + ".csv" for split in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )

    # saving options
    save_pkl(conf, save_opt)


def create_csv(
    save_folder, wav_lst, text_dict, split, select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "turnID", "duration", "agent", "previous_state", "wav", "current_state", "transcription"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in tzip(wav_lst):
        wav_file = wav_file[0]

        utterance_id = wav_file.replace(".wav", "")
        turn_id = int(utterance_id.split("/")[-1].split("-")[-1].split("_")[-1])
        previous_state = text_dict[utterance_id + "_previous"].replace("$", " dollars ")
        transcription = text_dict[utterance_id + "_transcription"].replace("$", " dollars ")
        agent = text_dict[utterance_id + "_agent"].replace("$", " dollars ")
        current_state = text_dict[utterance_id + "_current"].replace("$", " dollars ")

        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE
      
        if snt_cnt == select_n_sentences:
            break
        else:
            csv_line = [
                utterance_id,
                turn_id,
                str(duration),
                str(agent),
                str(previous_state),
                wav_file,
                str(current_state),
                str(transcription)
            ]

            # Appending current file to the csv_lines list
            csv_lines.append(csv_line)
            snt_cnt = snt_cnt + 1

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def check_multiwoz_folders(data_folder, version, splits):
    """
    Check if the data folder actually contains the Multi-Woz dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If Multi-Woz is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, "DSTC11_" + split)
        if not os.path.exists(split_folder):
            err_msg = (
                "The folder %s does not exist (it is expected in the "
                "Multi-Woz dataset)" % split_folder
            )
            raise OSError(err_msg)
