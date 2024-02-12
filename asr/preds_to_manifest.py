# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

import os
import argparse
from tqdm import tqdm

def manifest_to_txt(manifest_path: str, output_path: str, reference_path: str):
    """
    Converts the manifest which contains the predictions of an ASR model to the reference txt format.

    Arguments:
        - manifest_path: The name of the manifest file which contains the predicted transcriptions.
        - output_path: The path where to store the outputed predictions in the reference txt format.
        - reference_path: The path to the reference to extract the values which are not present in the manifest.
    """
    dialogues_preds = {}
    with open(manifest_path, "r") as manifest:
        for line in manifest:
            data = line.split(",", 1)
            dialogue_id = os.path.dirname(data[0]).split('/')[-1] + '.json'
            turn_id = os.path.basename(data[0]).split('.')[0].split('-')[1]
            dialogues_preds[dialogue_id + "_" + turn_id] = data[1]
    
    with open(output_path, "w") as out_txt, open(reference_path, "r") as in_txt:
        for line in tqdm(in_txt):
            splits = line.split(' ', 7)
            if splits[0] != 'END_OF_DIALOG\n':
                turn_id = int(splits[5])
                dialogue_id = splits[3]
                if turn_id % 2 == 1:
                    text = splits[-1]
                    if text.__contains__("state:"):
                        state = text.split("state:")[-1]
                    else:
                        state = "\n"
                    new_text = "user: " + dialogues_preds[dialogue_id + "_" + str(turn_id)].strip() + " state:" + state
                    out_txt.write(" ".join(splits[:7] + [new_text]))
                else:
                    out_txt.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=False,
                        help="The path to the dataset folder.", 
                        default="../data")
    parser.add_argument("--preds_folder", type=str, required=False,
                        help="The path to the predicted transcriptions folder.",
                        default="./results/1989/preds")
    args = parser.parse_args()

    splits = [
        "train",
        "dev_tts", "dev_human",
        "test_tts", "test_human"
    ]
    for split in splits:
        reference_manifest_path = os.path.join(args.data_folder, split.split("_")[0] + "_manifest.txt")
        predictions_path = os.path.join(args.preds_folder, split + ".csv")
        if split == "train":
            output_path = os.path.join(args.data_folder, split + "_tts_asr_manifest.txt")
        else:
            output_path = os.path.join(args.data_folder, split + "_asr_manifest.txt")
        
        print(f"Processing {predictions_path}...")
        manifest_to_txt(predictions_path, output_path, reference_manifest_path)

if __name__=="__main__":
    main()