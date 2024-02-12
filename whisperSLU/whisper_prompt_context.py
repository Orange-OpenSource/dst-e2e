# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

# import here because order of imports causes segfault
from torch.utils.tensorboard import SummaryWriter
import torch
import speechbrain as sb
import logging
import os
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import sys

from utils.evaluatePredictions import dialogueState_str2dict, cumulate

logger = logging.getLogger(__name__)

# Defining our dialogue utterance understanding model
class WhisperSLU(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.
        """

        batch = batch.to(self.device)
        wavs, wavs_lens = batch.sig
        outputs_tokens, outputs_lens = batch.outputs_tokens

        # Replacing the speechbrain padding tokens (id 0) with whisper tokenizer's padding id
        outputs_tokens = torch.where(outputs_tokens==0, self.tokenizer.pad_token_id, outputs_tokens).to(outputs_tokens.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wavs_lens)

        # Forward encoder + decoder
        encoder_out, logits, _ = self.modules.whisper(wavs, outputs_tokens)

        log_probs = self.hparams.log_softmax(logits)

        hyps = None

        if self.step % self.hparams.debug_print == 0:
            hyps = torch.argmax(log_probs, dim=-1)

        if stage != sb.Stage.TRAIN:
            # FIXME: only works for batches of 1 for valid since no middle padding for prompts
            prompt_tokens = batch.semantics_tokens[0][:,4:-1]
            prev_token = [self.tokenizer.encode('<|startofprev|>')[4]]
            prompt_tokens = prev_token + prompt_tokens.tolist()[0]
            self.hparams.valid_greedy_searcher.set_decoder_input_tokens(decoder_input_tokens=self.tokenizer.prefix_tokens, prompt=prompt_tokens)
            self.hparams.valid_greedy_searcher.set_language_token(self.tokenizer.prefix_tokens[1])
            # # This parameter is not the one used to set the max decoding steps, 
            # # so we rely on the decode ratio which is calculated from the shape of wav 
            # self.hparams.valid_greedy_searcher.max_length -= len(agent_tokens)
            with torch.no_grad():
                hyps, _ = self.hparams.valid_greedy_searcher(encoder_out.detach(), outputs_tokens)

        if stage == sb.Stage.VALID:
            
            loss = self.compute_objectives((log_probs, hyps, wavs_lens), batch, stage)
            
            # Logging the Valid Loss
            self.hparams.train_logger.writer.add_scalar(tag="Valid Loss", scalar_value=loss.detach().cpu(), 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step}, 
                                                        train_stats={"Valid Loss: ": loss.detach().cpu()})
            
            # Replacing the speechbrain padding tokens (id 0) with whisper tokenizer's padding id
            target_tokens, target_lens = batch.outputs_tokens_nobos
            target_tokens = torch.where(target_tokens==0, self.tokenizer.pad_token_id, target_tokens).to(target_tokens.device)

            # Writting the predictions in a file for future evaluation
            if not os.path.isdir(self.hparams.pred_folder):
                os.mkdir(self.hparams.pred_folder)
            with open(os.path.join(self.hparams.pred_folder, f'dev_{self.epoch}.csv'), "a") as pred_file:
                for hyp, element_id in zip(hyps, batch.id):
                    pred_file.write(f'{element_id},{self.tokenizer.decode(hyp)}\n')
            
            # Updating running accuracy
            self.accuracy.append(hyps, target_tokens)

        elif stage == sb.Stage.TEST:
            
            # Writting the predictions in a file for future evaluation
            with open(self.hparams.output_file, "a") as pred_file:
                for hyp, element_id in zip(hyps, batch.id):
                    pred_file.write(f'{element_id},{self.tokenizer.decode(hyp).replace(self.tokenizer.eos_token, "")}\n')

                    # Keeping track of the last predicted state of each dialogue to use it for the next prediction
                    if not self.hparams.gold_previous_state:
                        dialog_id = element_id.split("/")[-2]
                        turn_id = element_id.split("/")[-1].replace("Turn-", "")
                        json_state = dialogueState_str2dict(self.tokenizer.decode(hyp))

                        # Given that this approach is local, post processing predicted state
                        # See cumulate function for how this is done
                        previous_state = {}
                        if int(turn_id) > 1:
                            with open(os.path.join(self.hparams.output_folder, "last_turns", f'{dialog_id}.txt'), "r") as last_turn:
                                for line in last_turn:
                                    previous_state = dialogueState_str2dict(line)
                        json_state = cumulate(json_state, previous_state)
                        slots = []
                        for domain, slot_value in json_state.items():
                            slots.extend([f'{domain}-{slot}={value}' for slot, value in slot_value.items()])
                        state = "; ".join(slots)
                        with open(os.path.join(self.hparams.output_folder, "last_turns", f'{dialog_id}.txt'), "w") as last_turn:
                            last_turn.write(state + "\n")

        return log_probs, hyps, wavs_lens
    
    def compute_objectives(self, predictions, batch, stage):
        """
        Computes and returns the loss.
        """
        logprobs, hyps, wavs_lens = predictions
        batch = batch.to(self.device)
        outputs_tokens, outputs_lens = batch.outputs_tokens_nobos

        # Replacing the speechbrain padding tokens (id 0) with whisper tokenizer's padding id
        outputs_tokens = torch.where(outputs_tokens==0, self.tokenizer.pad_token_id, outputs_tokens).to(outputs_tokens.device)

        loss = self.hparams.nll_loss(
            logprobs, outputs_tokens, length=outputs_lens
        )

        return loss

    def fit_batch(self, batch):
        """
        Performs a forward and backward pass on a batch.
        """
        should_step = self.step % self.hparams.gradient_accumulation == 0
        debug_step = self.step % self.hparams.debug_print == 0

        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        (loss / self.hparams.gradient_accumulation).requires_grad_().backward()
        
        if should_step:
            
            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_step += 1

            # Update the learning rate
            old_lr, new_lr = self.hparams.lr_annealing(
                    self.optimizer
                )
            
            # Logging the loss and lr
            self.hparams.train_logger.writer.add_scalar(tag="Loss", scalar_value=loss.detach().cpu(), 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step}, 
                                                        train_stats={"Loss: ": loss.detach().cpu()})
            
            self.hparams.train_logger.writer.add_scalar(tag="LR", scalar_value=self.optimizer.param_groups[0]["lr"], 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step},
                                                train_stats={"LR": self.optimizer.param_groups[0]["lr"]})

        # Log the predictions and expected outputs for debug
        if debug_step:

            outputs_tokens, _ = batch.outputs_tokens_nobos
            semantics_tokens, _ = batch.semantics_tokens
            # Replacing the speechbrain padding tokens (id 0) with whisper tokenizer's padding id
            outputs_tokens = torch.where(outputs_tokens==0, self.tokenizer.pad_token_id, outputs_tokens).to(outputs_tokens.device)
            semantics_tokens = torch.where(semantics_tokens==0, self.tokenizer.pad_token_id, semantics_tokens).to(semantics_tokens.device)

            previous_states = self.tokenizer.batch_decode(semantics_tokens, skip_special_tokens=True)
            predictions = self.tokenizer.batch_decode(outputs[1], skip_special_tokens=False)
            targets = self.tokenizer.batch_decode(outputs_tokens, skip_special_tokens=True)

            log_text = "  \n  \n".join([f"Semantic input: {semantic}  \nPredicted output: {prediction}  \nExpected output: {target}\n"
                                        for semantic, prediction, target in zip(previous_states, predictions, targets)])
            self.hparams.train_logger.writer.add_text(tag="DEBUG-Train", text_string=log_text, 
                                                        global_step=self.optimizer_step)
            try:
                self.hparams.text_logger.log_stats(stats_meta = {"Step": self.optimizer_step}, 
                                                    train_stats={"DEBUG-Train": log_text})
            except UnicodeEncodeError:
                pass                  
        
        return loss.detach().cpu()

    def init_optimizers(self):
        """
        Initializing the optimizer.
        """
        model_params = self.modules.parameters()
        self.optimizer = self.hparams.opt_class(
            model_params
        )

    def on_stage_start(self, stage, epoch):

        if stage == sb.Stage.VALID:
            self.accuracy = self.hparams.acc_computer(self.tokenizer)
            if not os.path.isdir(self.hparams.pred_folder):
                os.mkdir(self.hparams.pred_folder)
            # Emptying file
            with open(os.path.join(self.hparams.pred_folder, f'dev_{self.epoch}.csv'), "w") as pred_file:
                pass

    def on_stage_end(self, stage, stage_loss, epoch):
        
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.epoch = epoch
        elif stage == sb.Stage.VALID:
            stage_stats["Accuracy"] = self.accuracy.summarize()
            self.hparams.train_logger.writer.add_scalar(tag="Dev Accuracy", scalar_value=stage_stats["Accuracy"], 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta = {"Step": self.optimizer_step}, 
                                            train_stats={"Dev Accuracy": stage_stats["Accuracy"]})

            self.checkpointer.save_checkpoint()
            sys.exit(42)
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """

        # Opening and closing the file to reset it
        with open(self.hparams.output_file, "w") as output:
            pass
        
        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )

    # Redefining fit for valid at every n_epochs_valid
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
        n_epochs_valid=1,
    ):

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            if epoch % n_epochs_valid == 0:
                self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break