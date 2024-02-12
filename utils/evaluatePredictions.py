# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

import argparse
from tqdm import tqdm
import copy
import os, json
import numpy as np

slot_types = ['attraction-area', 'attraction-name', 'attraction-type', 
              'hotel-area', 'hotel-name', 'hotel-type', 'hotel-day', 'hotel-people', 'hotel-pricerange', 
              'hotel-stay', 'hotel-stars', 'hotel-internet', 'hotel-parking',  
              'restaurant-area', 'restaurant-name', 'restaurant-food', 'restaurant-day', 'restaurant-people', 'restaurant-pricerange', 
              'restaurant-time',
              'taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat', 
              'train-arriveby', 'train-departure', 'train-destination', 'train-leaveat', 'train-people', 'train-day',
              'hospital-department',
              'bus-people', 'bus-leaveat', 'bus-arriveby', 'bus-day', 'bus-destination', 'bus-departure']

additional_slot_types = ['profile-name', 'profile-phonenumber', 'profile-idnumber', 'profile-email', 'profile-platenumber']

time_slots = ['restaurant-time', 'taxi-arriveby', 'taxi-leaveat', 'train-arriveby', 'train-leaveat']
open_slots = ['attraction-name', 'hotel-name', 'restaurant-name', 'taxi-departure', 'taxi-destination',
              'train-departure', 'train-destination']

# See https://github.com/AlibabaResearch/DAMO-ConvAI/issues/87 and listen to dialogues
spokenwoz_dialogues_synthesized = [
    # "SNG0601", "SNG0646", "SNG0653", "SNG0877", "SNG0885",
    # "SNG0890", "SNG0897", "SNG0901", "SNG0903"
    ]
    
def dialogueState_str2dict(dialogue_state: str, filtering=True):
    """
    Converts the ; separated Dialogue State linearization to a domain-slot-value dictionary.
    When *filtering* is set to True it filters out the slots which are not part of the ontology.
    """
    dict_state = {}

    # We consider every word after "[State] " to discard the transcription if present.
    dialogue_state = dialogue_state.split("[State] ")[-1]
    if ";" not in dialogue_state:
        return {}
    else:
        slots = dialogue_state.split(";")
        for slot_value in slots:
            if "=" not in slot_value:
                continue
            else:
                slot, value = slot_value.split("=")[0].strip(), slot_value.split("=")[1].strip()
                if filtering and slot not in slot_types and slot not in additional_slot_types:
                    continue
                elif "-" in slot:
                    domain, slot_type = slot.split("-")[0].strip(), slot.split("-")[1].strip()
                    if domain not in dict_state.keys():
                        dict_state[domain] = {}
                    dict_state[domain][slot_type] = value
            
        return dict_state
    
def dialogueState_dict2str(dialogue_state: dict):
    """
    Converts a dialogue state as a dictionnary per slot type per domain to a serielized ; seperated list
    """
    slots = []
    for domain, slots_values in dialogue_state.items():
        slots.extend([f"{domain}-{slot}={value}" for slot, value in slots_values.items()])
    
    return "; ".join(slots)
    
    
def cumulate(prediction, previous_state):
    """
    Cumulates the predicted state with the previous one in order to contextualize local Dialogue States.
    - <unk> is the value used to remove a slot
    - references are to previous concepts values are made through their slot type.
    """
    previous = copy.deepcopy(previous_state)
    for domain, slots_values in prediction.items():
        for slot, value in slots_values.items():
            if "<unk>" in value:
                # Suppressing the slot-value and the domain if no other slots
                if domain in previous:
                    if slot in previous[domain]:
                        del previous[domain][slot]
                    if previous[domain] == {}:
                        del previous[domain]
            
            elif value in slot_types or value in additional_slot_types:
                # Replacing the reference to a previous slot with its value
                # Ignoring the reference if incorrectly formated
                reference_value = ""
                try:
                    value_domain = value.split('-')[0]
                    value_slot = value.split('-')[1]
                    reference_value = previous[value_domain][value_slot]
                except:
                    #print(f"The reference to the slot {value} is incorrect.\nPrevious state: {previous}\nPrediction: {prediction}\n")
                    pass
                if domain not in previous:
                    previous[domain] = {}
                previous[domain][slot] = reference_value

            else:
                # Modifying or inserting a slot-value pair
                if domain not in previous:
                    previous[domain] = {}
                previous[domain][slot] = value
    
    return previous
    
class Metrics:

    def __init__(self, file, local=False, gold_previous=False, dataset="multiwoz", evaluate_ci=False):
        """
        Computes the Joint-Goal Accuracy (JGA) and Slot Precision and Recall for a given prediction file.
        """
        self.references = {}
        self.predictions = {}
        self.local = local
        self.file = file
        self.gold_previous = gold_previous
        self.dataset = dataset
        self.evaluate_ci = evaluate_ci

    def add_prediction(self, prediction, dialogue_id, turn_id, filtering=True):
        pred = dialogueState_str2dict(prediction, filtering)
        if self.local and str(turn_id) != "Turn-1":
            # Cumulate the current state with the previous one i.e. simple rule-base DST
            if self.gold_previous:
                pred = cumulate(pred, self.references[f'{dialogue_id}_Turn-{str(int(turn_id.split("-")[-1]) - 2)}'])
            else:
                pred = cumulate(pred, self.predictions[f'{dialogue_id}_Turn-{str(int(turn_id.split("-")[-1]) - 2)}'])
        if dialogue_id not in self.predictions:
            self.predictions[dialogue_id] = {}        
        self.predictions[dialogue_id][turn_id] = pred

    def add_reference(self, reference, dialogue_id, turn_id):
        if dialogue_id not in self.references:
            self.references[dialogue_id] = {}
        self.references[dialogue_id][turn_id] = dialogueState_str2dict(reference, filtering=False)
    
    def bootstrap_dialogues(self):
        """
        Samples, with replacement, N=size_of_dataset dialogues from the set of examples.
        """
        nbr_dialogues = len(self.dialogues)
        sample = np.random.choice(self.dialogues, nbr_dialogues)

        return sample
    
    def list_dialogues(self):
        self.dialogues = np.asarray([k for k in self.references.keys()])
    
    def prepare_samples(self, number_bootstrap_samples=1000, alpha=5):
        self.list_dialogues()
        if self.evaluate_ci:
            self.number_of_samples = number_bootstrap_samples
            self.samples = [self.bootstrap_dialogues() for _ in range(self.number_of_samples)]
            # Last sample is the dataset itself for mean computation
            self.samples.append(self.dialogues)
        else:
            self.number_of_samples = 0
            self.samples = [self.dialogues]
        self.alpha = alpha

    def slot_precision_recall(self):
        
        self.slot_scores = {}
        self.slot_value_scores = {}

        print("Computing Slot Precision Scores...")
        # for k, sampled_dialogues in tqdm(enumerate(self.samples)):
        k = self.number_of_samples
        sampled_dialogues = self.dialogues

        if self.dataset == "multiwoz":
            slot_list = slot_types
        elif self.dataset == "spokenwoz":
            slot_list = slot_types + additional_slot_types

        self.slot_scores[f"sample {k}"] = {slot_name: {'true-positive': 0,
                                                    'false-positive': 0,
                                                    'false-negative': 0}
                                                    for slot_name in slot_list}
        self.slot_value_scores[f"sample {k}"] = {slot_name: {'true-positive': 0,
                                                            'false-positive': 0,
                                                            'false-negative': 0}
                                                            for slot_name in slot_list}
        for dialogue_id in sampled_dialogues:
            for dialogue_turn, reference_state in self.references[dialogue_id].items():

                prediction = self.predictions[dialogue_id][dialogue_turn]

                for domain, slots in reference_state.items():
                    for slot, value in slots.items():
                        if domain in prediction:
                            if slot in prediction[domain]:
                                self.slot_scores[f"sample {k}"][f"{domain}-{slot}"]['true-positive'] += 1
                                if value == prediction[domain][slot]:
                                    self.slot_value_scores[f"sample {k}"][f"{domain}-{slot}"]['true-positive'] += 1
                                else:
                                    self.slot_value_scores[f"sample {k}"][f"{domain}-{slot}"]['false-negative'] += 1
                            else:
                                self.slot_scores[f"sample {k}"][f"{domain}-{slot}"]['false-negative'] += 1
                        else:
                            self.slot_scores[f"sample {k}"][f"{domain}-{slot}"]['false-negative'] += 1
            
            # Counting the false positives
            for dialogue_turn, predicted_state in self.predictions[dialogue_id].items():

                for domain, slots in predicted_state.items():
                    for slot, value in slots.items():
                        if f"{domain}-{slot}" not in self.slot_scores[f"sample {k}"]:
                            # slots which do not exist are ignored for scores
                            continue
                        elif domain in self.references[dialogue_id][dialogue_turn]:
                            if slot not in self.references[dialogue_id][dialogue_turn][domain]:
                                self.slot_scores[f"sample {k}"][f"{domain}-{slot}"]['false-positive'] += 1
                                self.slot_value_scores[f"sample {k}"][f"{domain}-{slot}"]['false-positive'] += 1
                        else:
                            self.slot_scores[f"sample {k}"][f"{domain}-{slot}"]['false-positive'] += 1

    def jga(self):

        self.jga_turn_scores = {}
        self.jga_scores = {}
        print("Computing Joint-Goal Accuracy...")
        for k, sampled_dialogues in enumerate(tqdm(self.samples)):
            # Joint-Goal Accuracy is computed per turn and (if needed) averaged over all turns.
            self.jga_turn_scores[f"sample {k}"] = {}

            for dialog_id in sampled_dialogues:

                for dialogue_turn, reference in self.references[dialog_id].items():
                    
                    prediction = self.predictions[dialog_id][dialogue_turn]
                    
                    if dialogue_turn not in self.jga_turn_scores[f"sample {k}"]:
                        self.jga_turn_scores[f"sample {k}"][dialogue_turn] = {"correct": 0, "total": 0} 
                    if prediction == reference:
                        self.jga_turn_scores[f"sample {k}"][dialogue_turn]["correct"] += 1
                    self.jga_turn_scores[f"sample {k}"][dialogue_turn]["total"] += 1
            
            total_correct = sum([turn["correct"] for _, turn in self.jga_turn_scores[f"sample {k}"].items()])
            total = sum([turn["total"] for _, turn in self.jga_turn_scores[f"sample {k}"].items()])
            self.jga_scores[f"sample {k}"] = round(100*total_correct/total, 1)

    def read_multiwoz_files(self, predictions_file: str, reference_manifest: str, filtering: bool):

        print("\nExtracting the references...\n")
        with open(reference_manifest, "r") as references:
            dialog_id = ""
            for line in tqdm(references):
                if line.__contains__("END_OF_DIALOG"):
                    pass
                else:
                    fields = line.split(' ', 7)
                    # A line looks like: line_nr: [N] dialog_id: [D.json] turn_id: [T] text: (user:|agent:) [ABC] state: domain1-slot1=value1; domain2-slot2=value2 
                    key_map = {"line_nr": 1, "dialog_id": 3, "turn_id": 5, "text": 7}
                    if fields[key_map["dialog_id"]].split(".json")[0] != dialog_id:
                        # Arriving on a new dialog we reset our dialogue_id
                        dialog_id = fields[key_map["dialog_id"]].split(".json")[0]
                    turn_id = fields[key_map["turn_id"]]

                    # User turn line
                    if int(turn_id) % 2 == 1:                    
                        # Extracting the text part (transcription and state) of the line
                        text_split = fields[key_map["text"]].split("state:")
                        state = text_split[-1].strip()
                        self.add_reference(state, dialogue_id=dialog_id, turn_id=f'Turn-{turn_id}')

        print("\nExtracting the predictions...\n")
        with open(predictions_file, "r") as predictions:
            for line in tqdm(predictions):
                # The predictions csv is composed of the id and the prediction
                fields = line.split(',', 1)
                dialogue_id = fields[0].split('/')[-2]
                turn_id = fields[0].split('/')[-1]
                self.add_prediction(fields[1].strip(), dialogue_id=dialogue_id, turn_id=turn_id, filtering=filtering)

        if self.references.keys() != self.predictions.keys():
            raise AssertionError(f"Carefull the predictions ({predictions_file}) and references ({reference_manifest}) do not concern strictly the same set of examples.")

    def read_spokenwoz_files(self, predictions_file: str, reference_manifest: str, filtering: bool):

        with open(reference_manifest, "r") as data:
            for line in data:
                annotations = json.loads(line)
        
        if "dev" in reference_manifest:
            # Selecting only dialogues for dev set
            dev_ids = []
            folder = os.path.dirname(reference_manifest)
            with open(os.path.join(folder, "valListFile.json"), "r") as val_list_file:
                for line in val_list_file:
                    dev_ids.append(line.strip())
            dialogues_to_remove = [k for k, _ in annotations.items() if k not in dev_ids]
            for dialog_id in dialogues_to_remove:
                del annotations[dialog_id]
        
        for dialog_id, dialog_info in annotations.items():
            for turn_id, turn_info in enumerate(dialog_info["log"]):
                if turn_id % 2 == 1:
                    # Dialogue States annotations are on Agent turns 
                    state = []
                    for domain, info in turn_info["metadata"].items():
                        for slot, value in info["book"].items():
                            if slot != "booked" and value != '':
                                state.append(f'{domain}-{slot}={value}')
                        for slot, value in info["semi"].items():
                            if value != "":
                                # One example in train set has , between numbers
                                state.append(f'{domain}-{slot}={value.replace(",", "")}')
                    if dialog_id not in spokenwoz_dialogues_synthesized:
                        self.add_reference("; ".join(state), dialogue_id=dialog_id, turn_id=f'Turn-{turn_id-1}')
        
        print("\nExtracting the predictions...\n")
        with open(predictions_file, "r") as predictions:
            for line in tqdm(predictions):
                # The predictions csv is composed of the id and the prediction
                fields = line.split(',', 1)
                dialogue_id = fields[0].split('/')[-2]
                turn_id = fields[0].split('/')[-1]
                # Example of id: SNG1751_Turn-26
                dialogue_id = fields[0].split('/')[-1].split('_')[0]
                turn_id = fields[0].split('/')[-1].split('_')[1]
                if dialogue_id not in spokenwoz_dialogues_synthesized:
                    self.add_prediction(fields[1].strip(), dialogue_id=dialogue_id, turn_id=turn_id, filtering=filtering)

        if self.references.keys() != self.predictions.keys():
            raise AssertionError(f"Carefull the predictions ({predictions_file}) and references ({reference_manifest}) do not concern strictly the same set of examples.")


    def summary(self):
        
        self.jga()
        self.slot_precision_recall()
        summary = f"==================Metric report of {self.file}==================\n"
        
        evaluated_jga = self.jga_scores[f"sample {self.number_of_samples}"]
        if self.evaluate_ci:
            # Get confidence interval over values
            # https://github.com/luferrer/ConfidenceIntervals/blob/main/confidence_intervals/confidence_intervals.py#L165
            jga_values = [jga for sample, jga in self.jga_scores.items() if sample != f"sample {self.number_of_samples}"]
            jga_low = np.percentile(jga_values, self.alpha/2)
            jga_high = np.percentile(jga_values, 100-self.alpha/2)
            summary += f'Joint-Goal Accuracy = {evaluated_jga}% ({jga_low}, {jga_high})\n'
        else:
            summary += f'Joint-Goal Accuracy = {evaluated_jga}%\n'
        
        evaluated_per_turn_jga = {turn: round(100*values["correct"]/values["total"], 1) 
                                  for turn, values in self.jga_turn_scores[f"sample {self.number_of_samples}"].items()}
        if self.evaluate_ci:
            per_turn_jga_values = {turn : [] for turn in self.jga_turn_scores[f"sample {self.number_of_samples}"].keys()}
            for sample, per_turn_jga in self.jga_turn_scores.items():
                if sample != f"sample {self.number_of_samples}":
                    for turn, stats in per_turn_jga.items():
                        per_turn_jga_values[turn].append(100*stats["correct"]/stats["total"])
            low_per_turn_jga = {turn: np.percentile(values, self.alpha/2) for turn, values in per_turn_jga_values.items()}
            high_per_turn_jga = {turn: np.percentile(values, 100-self.alpha/2) for turn, values in per_turn_jga_values.items()}
            turn_cis = [(value, low_per_turn_jga[turn], high_per_turn_jga[turn]) 
                        for turn, value in evaluated_per_turn_jga.items()]
            summary += "\tPer-turn: \n\t\t" + f'{turn_cis}\n\n'
        else:
            summary += "\tPer-turn: \n\t\t" + f'{[value for value in evaluated_per_turn_jga.values()]}\n\n'
        
        open_f1s = []
        time_f1s = []
        cat_f1s = []

        summary += "Slot Values Scores:\n"
        for slot, scores in self.slot_value_scores[f"sample {self.number_of_samples}"].items():
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

        summary += f"\t- Open slots:\n"
        summary += f'\t\t- F1s: {open_f1s}\n'
        summary += f"\t- Time slots:\n"
        summary += f'\t\t- F1s: {time_f1s}\n'
        summary += f"\t- Categorical slots:\n"
        summary += f'\t\t- F1s: {cat_f1s}\n'
        
        return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_manifest", type=str, help="The path to the reference txt file.",
                        default="../data/dev_manifest.txt")
    parser.add_argument("--predictions", type=str, help="The path where to find the csv file with the models predictions.")
    parser.add_argument("--local", action='store_true', default=False,
                        help="Considers the predicted states as turn level predictions hence contextualizes them.")
    parser.add_argument("--no_filtering", action='store_true', default=False,
                        help="Deactivates the slot ontology predictions filtering.")
    parser.add_argument("--previous_gold", action='store_true', default=False,
                        help="Cumulates with the previous ground-truth instead of the previous prediction. Only used with the local argument.")
    parser.add_argument("--dataset", type=str, default="multiwoz",
                        help='The dataset ("multiwoz" or "spokenwoz") to choose the dataformat reading.')
    parser.add_argument("--evaluate_ci", action='store_true', default=False,
                        help="Whether to evaluate the confidence intervals of the JGA.")
    args = parser.parse_args()

    metrics = Metrics(file=args.predictions, local=args.local, gold_previous=args.previous_gold,
                      dataset=args.dataset, evaluate_ci=args.evaluate_ci)
    
    if args.dataset == "multiwoz":
        metrics.read_multiwoz_files(predictions_file=args.predictions, reference_manifest=args.reference_manifest, 
                                    filtering=not args.no_filtering)
    elif args.dataset == "spokenwoz":
        metrics.read_spokenwoz_files(predictions_file=args.predictions, reference_manifest=args.reference_manifest, 
                                    filtering=not args.no_filtering)
    else:
        parser.error('Argument dataset should be either "multiwoz" or "spokenwoz".')
    
    metrics.prepare_samples()

    print(metrics.summary())

if __name__=="__main__":
    main()