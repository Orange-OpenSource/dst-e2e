# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

"""
Calculate Joint Goal Accuracy. 
Reference: https://aclanthology.org/P18-1135.pdf
"""
from utils.evaluatePredictions import dialogueState_str2dict

def JGA(predictions, targets, tokenizer):
    """Calculates the Joint-Goal Accuracy for predicted tokens and target tokens in a batch.

    Arguments
    ----------
    predictions : tensor
        Predicted tokens (batch_size, sequence length).
    targets : tensor
        Target (batch_size, sequence length).
    tokenizer : Tokenizer
        Tokenizer with a decode method associated with the model.
    """
    correct = 0
    total = 0

    for prediction, target in zip(predictions, targets):
        # Decoding the tokens to text
        prediction_txt = tokenizer.decode(prediction)
        target_txt = tokenizer.decode(target)
        # Removing the eos ("</s>") token and converting the dialogue state ; separated list to a dict
        prediction_dict = dialogueState_str2dict(prediction_txt.replace(tokenizer.eos_token, ""))
        reference_dict = dialogueState_str2dict(target_txt.replace(tokenizer.eos_token, ""))
        if prediction_dict == reference_dict:
            correct += 1
        total += 1
     
    return float(correct), float(total)


class AccuracyStats:
    """
    Module to track and compute the Joint-Goal Accuracy.

    Arguments
    ----------
    tokenizer : Tokenizer
        Tokenizer with decode method
    """

    def __init__(self, tokenizer):
        self.correct = 0
        self.total = 0
        self.tokenizer = tokenizer

    def append(self, predictions, targets):
        """
        This function is for updating the stats according to the prediction
        and target in the current batch.

        Arguments
        ----------
        predictions : tensor
            Predicted tokens (batch_size, sequence length).
        targets : tensor
            Target tokens (batch_size, sequence length).
        """
        numerator, denominator = JGA(predictions, targets, self.tokenizer)
        self.correct += numerator
        self.total += denominator

    def summarize(self):
        """
        Averages the current Joint-Goal Accuracy (JGA).
        """
        return round(100*self.correct / self.total, 2) if self.total != 0 else 100.00
