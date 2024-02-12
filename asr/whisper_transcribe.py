# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

#!/usr/bin/env python3

import os
import json
import argparse
import torch
from tqdm import tqdm

from  speechbrain.processing.speech_augmentation import Resample
from speechbrain.dataio.dataio import read_audio

import whisper

def transcribe_split(dataset: str, data_folder: str, split:str, whisper_model, output_folder:str):
    """
    Creates a copy of the split manifest (audio linked to annotations) 
    with the transcriptions from whisper model.
    """    
    # Decoding options
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    if dataset == "multiwoz":
        reference_manifest_path = os.path.join(data_folder, split.split("_")[0] + "_manifest.txt")
        if split == "train":
            split = "train_tts"
            output_path = os.path.join(output_folder, split + "_whisper_manifest.txt")
        else:
            output_path = os.path.join(output_folder, split + "_whisper_manifest.txt")
        
        with open(output_path, "w") as out_txt, open(reference_manifest_path, "r") as in_txt:
            for line in tqdm(in_txt):
                splits = line.split(' ', 7)
                if splits[0] != 'END_OF_DIALOG\n':
                    turn_id = int(splits[5])
                    dialog_id = splits[3].replace(".json", "")
                    if turn_id % 2 == 1:
                        text = splits[-1]
                        state = text.split("state:")[-1]

                        # Compute transcription 
                        audio_file = os.path.join(data_folder, f"DSTC11_{split}", dialog_id, f"Turn-{turn_id}.wav")
                        #audio = whisper.load_audio(audio_file)
                        audio = torch.squeeze(read_audio(audio_file))
                        audio = whisper.pad_or_trim(audio)
                        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
                        transcription = whisper.decode(whisper_model, mel, options).text

                        new_text = "user: " + transcription.strip() + " state:" + state
                        out_txt.write(" ".join(splits[:7] + [new_text]))
                    else:
                        out_txt.write(line)
    
    elif dataset == "spokenwoz":
        
        reference_path = os.path.join(data_folder, f"text_5700_{split}", "data.json")
        output_path = os.path.join(output_folder, f"{split}_data_whisper.json")
        
        with open(reference_path, "r") as reference:
            for line in reference:
                annotations = json.loads(line)

        # Adding whisper's transcriptions to the data
        SAMPLERATE = 8000
        for dialog_id, dialog_info in tqdm(annotations.items()):
            
            # Reading dialogue audio
            audio_file = os.path.join(data_folder, f"audio_5700_{split}", f"{dialog_id}.wav")
            resampler = Resample(orig_freq=8000, new_freq=16000)
            audio = read_audio(audio_file)
            audio = audio.unsqueeze(0) # Must be B*T*C
            resampled = resampler(audio)
            # Fusing both channels
            resampled = torch.mean(resampled, dim=2)

            for turn_id, turn_info in enumerate(dialog_info["log"]):
                end_time = turn_info["words"][-1]["EndTime"]*(SAMPLERATE//1000)
                if turn_id == 0:
                    start_time = 0
                else:
                    start_time = turn_info["words"][0]["BeginTime"]*(SAMPLERATE//1000)

                # Compute transcription
                # Selecting the correct frames: start*2 bc resampled
                # FIXME: not sure if should transcribe whole dialog and match timestamps of transcribe turn per turn
                # Adding 1s of blank before and after turn to help transcribe
                audio_turn = torch.squeeze(resampled)[int(start_time)*2:int(end_time)*2]
                blank = torch.zeros(50)
                audio_turn = torch.cat((blank, audio_turn, blank))
                # FIXME: pad_or_trim cuts over 30s
                audio_turn = whisper.pad_or_trim(audio_turn)
                mel = whisper.log_mel_spectrogram(audio_turn).to(whisper_model.device)
                transcription = whisper.decode(whisper_model, mel, options).text

                turn_info["whisper"] = transcription.strip()
        
        # Writing in the output
        with open(output_path, "w") as output:
            output.write(json.dumps(annotations))             
    
    else:
        print("Did nothing since only supposed to transcribe for multiwoz or spokenwoz datasets.")
    
    return 
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=False,
                        help="The path to the dataset folder.", 
                        default="../data")
    dataset = parser.add_argument("--dataset", type=str, required=False,
                        help="The dataset (multiwoz or spokenwoz) to transcribe.",
                        default="multiwoz")
    parser.add_argument("--model_size", type=str, required=False,
                        help="The size of the desired whisper model.",
                        default="small.en")
    parser.add_argument("--output_folder", type=str, required=False,
                        default="../data")
    args = parser.parse_args()

    if args.dataset == "multiwoz":
    
        splits = [
            "train",
            "dev_tts", "dev_human",
            "test_tts", "test_human"
        ]
    
    elif args.dataset == "spokenwoz":
        
        splits = [
            "train_dev",
            "test"
        ]
    
    else:
        raise argparse.ArgumentError(dataset, 'Should be one of "multiwoz" or "spokenwoz"')

    # load model and processor
    whisper_model = whisper.load_model(args.model_size)

    # Perform transcriptions
    for split in splits:
        transcribe_split(dataset=args.dataset, data_folder=args.data_folder,
                        split=split, whisper_model=whisper_model, 
                        output_folder=args.output_folder)

if __name__=="__main__":
    main()