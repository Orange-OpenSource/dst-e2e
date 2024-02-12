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

from utils.evaluatePredictions import Metrics

def build_plot(data, dataset: str, turn_threshold: int):
    """
    Builds a line plot with 
        - Joint Goal Accuracy as x-axis
        - Turn number as y-axis
        - Approach used as color
        - Usage of ground truth previous state as shape 
    """
    dfs = []
    for approach, method_jga in data.items():

        # Build a dataframe for each approach
        approach_turn_accuracies_gold = [[1 + turn*2, acc] for turn, acc in enumerate(method_jga["ground-truth"])]
        approach_turn_accuracies_pred = [[1 + turn*2, acc] for turn, acc in enumerate(method_jga["prediction"])]
        
        approach_gold_df = pd.DataFrame(approach_turn_accuracies_gold, columns=["Turn", "JGA"])
        kwargs = {"Previous State": lambda x: "Ground-truth"}
        approach_gold_df = approach_gold_df.assign(**kwargs)
        approach_gold_df = approach_gold_df.assign(Approach=lambda df: approach)

        approach_pred_df = pd.DataFrame(approach_turn_accuracies_pred, columns=["Turn", "JGA"])
        kwargs = {"Previous State": lambda x: "Prediction"}
        approach_pred_df = approach_pred_df.assign(**kwargs)
        approach_pred_df = approach_pred_df.assign(Approach=lambda df: approach)

        dfs.append(pd.concat([approach_gold_df, approach_pred_df], axis=0).reset_index(drop=True))

    # Combining all approaches in one dataframe
    df = pd.concat(dfs, axis=0).reset_index(drop=True)

    sns.set_style("darkgrid")

    # Caping the analysis to 31 (50 for spokenwoz) turns as there are fewer and fewer dialogues with n turns as n increases
    lineplot = sns.lineplot(data=df[df["Turn"] < turn_threshold], x="Turn", y="JGA", hue="Approach", style="Previous State", markers=True)
    lineplot.set(xticks=[1 + 2*x for x in range(turn_threshold//2)])
    sns.move_legend(lineplot, "lower left")
    lineplot.get_figure().savefig(f"{dataset}_turn_accuracy_test_Human.pdf")


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
        turn_threshold = 30
    elif args.dataset == "spokenwoz":
        reference_file = os.path.join(args.data_folder, "text_5700_test", "data.json")
        turn_threshold = 50
    else:
        parser.error('Please select a dataset between "multiwoz" and "spokenwoz".')
    
    data = {}
    for method in methods:

        data[method] = {}

        # Computing the per-turn JGA of each method
        pred_file = os.path.join(args.root_output_folder, f"dstE2E_{args.dataset}_{method_version_mapping[method]}_0.000100", "preds", "test_human.csv")
        metrics = Metrics(file=pred_file, local=method=="Local", dataset=args.dataset)

        if args.dataset == "multiwoz":
            metrics.read_multiwoz_files(predictions_file=pred_file, reference_manifest=reference_file, filtering=True)
        elif args.dataset == "spokenwoz":
            metrics.read_spokenwoz_files(predictions_file=pred_file.replace("_human", ""), 
                                        reference_manifest=reference_file, filtering=True)
        
        metrics.prepare_samples()
        metrics.summary()

        per_turn_jga = {turn: round(100*values["correct"]/values["total"], 2) 
                        for turn, values in metrics.jga_turn_scores[f"sample {metrics.number_of_samples}"].items()}
        data[method]["ground-truth"] = [value for value in per_turn_jga.values()]

        # Same with the predicted previous state
        previous_pred_file = pred_file.replace(".csv", "_previous.csv")
        pred_metrics = Metrics(file=previous_pred_file, local=method=="Local", dataset=args.dataset)

        if args.dataset == "multiwoz":
            pred_metrics.read_multiwoz_files(predictions_file=previous_pred_file, reference_manifest=reference_file, filtering=True)
        elif args.dataset == "spokenwoz":
            pred_metrics.read_spokenwoz_files(predictions_file=previous_pred_file.replace("_human", ""), 
                                            reference_manifest=reference_file, filtering=True)
        
        pred_metrics.prepare_samples()
        pred_metrics.summary()

        pred_per_turn_jga = {turn: round(100*values["correct"]/values["total"], 2) 
                            for turn, values in pred_metrics.jga_turn_scores[f"sample {pred_metrics.number_of_samples}"].items()}
        data[method]["prediction"] = [value for value in pred_per_turn_jga.values()]

    build_plot(data, args.dataset, turn_threshold)
