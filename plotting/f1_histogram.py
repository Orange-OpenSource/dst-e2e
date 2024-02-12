# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

import seaborn as sns
import pandas as pd
import argparse
import os

from utils.evaluatePredictions import Metrics, time_slots, open_slots

def build_plot(f1s, dataset: str, add_legend: bool):
    """
    Builds a bar plot with 
        - color encoding for the approach
        - the x axis for the slot group 
        - the y axis the average F1
    """
    dfs = []
    for method, slot_category_f1s in f1s.items():

        method_f1s = []
        for category, values in slot_category_f1s.items():
            method_f1s.extend([[category, value] for value in values])
        method_df = pd.DataFrame(method_f1s, columns=["Slot Group", "F1"])
        dfs.append(method_df.assign(Approach=lambda df: f"{method}"))

    df = pd.concat(dfs, axis=0).reset_index()

    sns.set_style("darkgrid")
    barplot = sns.barplot(data=df, x="Slot Group", y="F1", hue="Approach", 
                        errorbar=None, #('ci', 5), n_boot=1000, 
                        legend=add_legend, capsize=.1)
    barplot.set_ylim(bottom=0.8, top=1)
    if add_legend:
        sns.move_legend(barplot, "lower left")
    barplot.get_figure().savefig(f"{dataset}_slot_group_F1_test_human.pdf")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_output_folder", type=str, help="The path to the folder where to find all methods.",
                        default="../results")
    parser.add_argument("--data_folder", type=str, help="Path to the dataset manifest files.",
                        default="../data")
    parser.add_argument("--dataset", type=str, help="dataset for which to plot the histogram.",
                        default="multiwoz")
    args = parser.parse_args()

    method_version_mapping = {
        "Cascade-WavLM": "cascade-nlu",
        "Cascade-Whisper": "cascade-whisper",
        "Local": "local", 
        "E2E-WavLM":"global",
        "E2E-Whisper": "global-whisper"
        }

    methods = [
        "Cascade-WavLM", 
        "Cascade-Whisper", 
        # "Local", 
        "E2E-WavLM",
        "E2E-Whisper"
        ]

    if args.dataset == "multiwoz":
        reference_file = os.path.join(args.data_folder, "test_manifest.txt")
        add_legend = False
    elif args.dataset == "spokenwoz":
        reference_file = os.path.join(args.data_folder, "text_5700_test", "data.json")
        add_legend = True
    else:
        parser.error('Please select a dataset between "multiwoz" and "spokenwoz".')

    data = {}
    for method in methods:
        
        # Computing the per-slot f1 of each method
        pred_file = os.path.join(args.root_output_folder, f"dstE2E_{args.dataset}_{method_version_mapping[method]}_0.000100", "preds", "test_human.csv")
        metrics = Metrics(file=pred_file, local=method=="Local", dataset=args.dataset)
        
        if args.dataset == "multiwoz":
            metrics.read_multiwoz_files(predictions_file=pred_file, reference_manifest=reference_file, filtering=True)
        elif args.dataset == "spokenwoz":
            metrics.read_spokenwoz_files(predictions_file=pred_file.replace("_human", ""), reference_manifest=reference_file, filtering=True)

        metrics.prepare_samples()
        metrics.summary()

        open_f1s = []
        time_f1s = []
        cat_f1s = []

        for slot, scores in metrics.slot_value_scores[f"sample {metrics.number_of_samples}"].items():
            tp = scores["true-positive"]
            fp = scores["false-positive"]
            fn = scores["false-negative"]
            precision = tp/(tp + fp) if (tp + fp) != 0 else 1
            recall = tp/(tp + fn) if (tp + fn) != 0 else 1
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) != 0 else 1
            if slot in open_slots:
                open_f1s.append(f1)
            elif slot in time_slots:
                time_f1s.append(f1)
            else:
                cat_f1s.append(f1)

            data[method] = {
                "Categorical": cat_f1s,
                "Non-Categorical": open_f1s,
                "Time": time_f1s
            }
    
    build_plot(data, args.dataset, add_legend)
