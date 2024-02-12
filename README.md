# Spoken Dialogue State Tracking

This repository presents the code associated with the paper ["Is one brick enough to break the wall of spoken dialogue state tracking?"](https://hal.science/hal-04267804). 

## General overview

Traditionally, Task-Oriented Dialogue systems update their understanding of the user's needs in three steps: transcription of the user's utterance, semantic extraction of the key concepts, and contextualization with the previously identified concepts. Such cascade approaches suffer from cascading errors and separate optimization. End-to-End approaches have been proved helpful up to the semantic extraction step. This repository attempts to go one step further paving the path towards completely neural spoken dialogue state tracking by comparing three approaches: (1) a state of the art cascade approach, (2) a locally E2E approach with rule-based contextualization and (3) a completely neural approach.

## Getting started

- [ ] Clone this repository with `git clone ...`
- [ ] Create a new conda environment with `conda create -n myenv python=3.10`.
- [ ] Enter your fresh environment with `conda activate myenv`
- [ ] Run `pip install -r requirements.txt`.
- [ ] Run `pip install -e .`.
- [ ] Download [the datasets](#datasets) 
    - [ ] spoken MultiWoz with `cd data && bash download_spoken_MultiWoz.sh`.
    - [ ] Extract the audio from each split with `cd ../utils && python extract_audio.py` (use `--data_folder YOUR_DATA_FOLDER` for a personal data folder).
    - [ ] SpokenWoz with `cd data && bash download_spokenWoz.sh`.
- [ ] Run the [training scripts](#training).
- [ ] Run the [evaluations](#evaluation).

### Datasets

#### Spoken MultiWoz

The spoken MultiWoz dataset, adapted from MultiWoz 2.1, is available on the [Speech Aware Dialogue System Technology Challenge website](https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html). It consists of audio recordings of user turns and mappings to integrate them with the written agent turns in order to form a full dialogue. Please place all the files in a common folder and rename the manifest files `[split]_manifest.txt` and the split folders `DSTC11_[split]_[vocalization]/`.

- Training data (TTS):
    - [train.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.zip) contains 4 subdirectories, one for each TTS speaker (tpa, tpb, tpc, tpd), and each subdirectories contains all the 8434 dialogs corresponding to the original training set. The TTS outputs were generated using speakers that are available via Google Cloud Speech API.
    - [train.tts-verbatim.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt) contains the original dialog training data, which is used to generate the TTS outputs.
- Dev data (TTS):
    - [dev-dstc11.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.tts-verbatim.2022-07-27.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [dev-dstc11.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-07-27.txt) contains the mapping from user utterances back to the original dialog.
- Dev data (Human):
    - [dev-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data (TTS):
    - [test-dstc11-tts-verbatim.2022-09-21.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11-tts-verbatim.2022-09-21.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [test-dstc11.2022-09-21.txt](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.2022-09-21.txt) contains the mapping from user utterances back to the original dialog.
- Test data (Human):
    - [test-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data DST annotations:
    - [test-dstc11.2022-1102.gold.json](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-1102.gold.json) contains the gold DST annotations for the test set.
    - We provide a [test manifest](data/test_manifest.txt) with the DST annotations integrated in the same format as the other splits.

#### SpokenWoz

The SpokenWoz dataset is available on their [official website](https://spokenwoz.github.io/SpokenWOZ-github.io/). It consists of human-human task-oriented dialogue recordings associated with Dialogue States for each agent dialogue turn. 

- Train and Dev data:
    - [audio_5700_train_dev.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz) contains each dialogue's audio recording.
    - [text_5700_train_dev.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz) contains the list of dialogues to consider for the dev split in `valListFile.json` and the annotations for each dialogue turn in `data.json`.
- Test data:
    - [audio_5700_test.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz) contains each dialogue's audio recording.
    - [text_5700_test.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz) contains the annotations for each dialogue turn in `data.json`.

### Training

Here is how to train the three approaches:

- For the cascade system
    - Train the Automatic Speech Recognition system (only MultiWoz) by running `python asr/train_asr.py asr/hparams/train_asr.yaml --data_folder PATH_TO_DATASET`.
        - The script will also provide predictions over the training set at `./asr/results/1989/preds/`.
        - Convert the outputs into a manifest file by running `python asr/preds_to_manifest.py --data_folder PATH_TO_DATASET`.
    - Run Whisper transcription with `python asr/whisper_transcribe.py --dataset multiwoz --data_folder PATH_TO_DATASET --output_folder PATH_TO_DATASET`.
    - Train the Dialogue State Tracking model over the training set transcriptions by running `python w2v+t5/train_multiwoz.py w2v+t5/hparams/train_multiwoz.yaml --data_folder PATH_TO_DATASET --version [cascade-nlu|cascade-whisper]`.
- For the local system
    - Run `python whisperSLU/train_multiwoz.py whisperSLU/hparams/train_multiwoz.yaml --data_folder PATH_TO_DATASET`.
- For the completely neural system
    - Run `python w2v+t5/train_multiwoz.py w2v+t5/hparams/train_multiwoz.yaml --data_folder PATH_TO_DATASET`.
    - To use whisper encoder instead of WavLM switch to the hparams file `w2v+t5/hparams/train_multiwoz_with_whisper_enc.yaml`.

Note that, apart from the ASR training and the local approach, you can replace "multiwoz" by "spokenwoz" to run the same operations on the SpokenWoz dataset. You can also add `--inference True` to perform Dialogue State inference with any of those methods and `--gold_previous_state False` to infer the next state from the previous predicted one.

### Evaluation

Evaluate your predictions in terms of Joint-Goal Accuracy (at turn and dialogue level) and Slot Precision (per slot groups) with the script [evaluatePredictions.py](utils/evaluatePredictions.py).

```
python evaluatePredictions.py --reference_manifest PATH_TO_SPLIT_MANIFEST --predictions PATH_TO_PREDICTIONS_CSV --dataset [multiwoz|spokenwoz]
```

To evaluate the 95% confidence intervals of the JGA scores, with a bootstrapping strategy, add the argument `--evaluate_ci`. 

For the local approach, use the `--local` option to consider the predictions as local dialogue states and solve cross turn references and `--previous_pred` to perform this resolution with the previously predicted state instead of the ground truth.

Please feel free to explore the several parameters to play with in the `hparams/*.yaml` files and within each script.

The expected Joint-Goal Accuracy (JGA) results are reported in the following tables: 

#### Spoken Multi-Woz

|   Gold Previous State    |>|     Dev   |>|   Test     |
|                          | TTS | Human | TTS  | Human |
|:------------------------:|:---:|:-----:|:----:|:-----:|
| Cascade (WavLM)          | 58.2| 55.0  | 57.2 | 53.5  |
| Cascade (Whisper)        | 63.7| 63.6  | 64.4 | 62.3  |
| Local                    | 42.7| 41.4  | 41.7 | 40.8  |
| Global (WavLM)           | 56.4| 54.0  | 53.4 | 53.0  |
| Global (Whisper)         | 59.0| 56.9  | 58.3 | 56.6  |

| Predicted Previous State |>|    Dev    |>|    Test    |
|                          | TTS | Human | TTS  | Human |
|:------------------------:|:---:|:-----:|:----:|:-----:|
| Cascade (WavLM)          | 19.5| 16.2  | 17.6 | 15.3  |
| Cascade (Whisper)        | 24.0| 21.9  | 23.1 | 21.3  |
| Local                    | 13.0| 11.3  | 12.4 | 11.2  |
| Global (WavLM)           | 15.1| 14.4  | 13.7 | 14.6  |
| Global (Whisper)         | 19.1| 17.6  | 18.5 | 16.6  |

#### SpokenWoz

See [issue](https://github.com/AlibabaResearch/DAMO-ConvAI/issues/87) for missaligned audio files in the test split. The audio was ignored when not available.

|   Gold Previous State    |     Dev     |     Test     |
|:------------------------:|:-----------:|:------------:|
| Cascade (their ASR)      |     82.3    |     63.0     |
| Cascade (Whisper)        |     80.7    |     64.2     |
| Global (WavLM)           |     70.7    |     61.8     |
| Global (Whisper)         |     81.6    |     80.5     |

| Predicted Previous State |     Dev     |     Test     |
|:------------------------:|:-----------:|:------------:|
| Cascade (their ASR)      |     24.6    |     23.4     |
| Cascade (Whisper)        |     24.3    |     23.5     |
| Global (WavLM)           |     22.2    |     20.3     | 
| Global (Whisper)         |     26.5    |     24.1     |

## Acknowledgements

The authors would like to thank the [SpeechBrain](https://speechbrain.github.io/) team for the [ASR recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice/ASR/CTC) and their great toolkit in general.

When using this code please cite this paper:

```
@inproceedings{druart2024e2edst,
    title={Is one brick enough to break the wall of spoken dialogue state tracking?},
    author={Druart, Lucas and Vielzeuf, Valentin and Estève, Yannick},
    booktitle={},
    year={2024}
}
```

# License

Copyright (c) 2024 Orange

This code is released under the MIT license. See the [LICENSE](LISCENCE) file for more information.
