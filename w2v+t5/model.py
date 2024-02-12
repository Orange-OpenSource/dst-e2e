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
import sys
import torch
import speechbrain as sb

# Profiling
from tqdm.contrib  import tqdm

import os
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import itertools
import logging

from utils.evaluatePredictions import dialogueState_str2dict, dialogueState_dict2str

logger = logging.getLogger(__name__)


# Defining our Dialogue model
class DialogueUnderstanding(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """
        Forward computation of the model involves a forward pass on both
        the audio and semantic encoders and a forward pass on the decoder.
        """

        batch = batch.to(self.device)
        wavs, wavs_lens = batch.sig
        semantics_tokens, semantics_lens = batch.semantics_tokens
        outputs_tokens, outputs_lens = batch.outputs_tokens

        # Replacing the speechbrain padding tokens (id 0) with T5 tokenizer's padding id
        # Careful since the bos token in output_tokens is a padding token!
        semantics_tokens = torch.where(semantics_tokens==0, self.tokenizer.pad_token_id, semantics_tokens)
        outputs_tokens_nobos = torch.where(outputs_tokens==0, self.tokenizer.pad_token_id, outputs_tokens)[:,1:]
        outputs_tokens = torch.cat((torch.zeros_like(outputs_tokens[:,0]).unsqueeze(-1), outputs_tokens_nobos), axis=1).to(outputs_tokens.device)        

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wavs_lens)

        semantics_enc_out = self.modules.semantic_encoder(semantics_tokens, semantics_lens)
        
        # Given that some SpokenWoz files are falsly segmented, identifying the dialogues concerned with the except
        try:
            if self.hparams.version == "cascade-nlu" or self.hparams.version == "cascade-whisper":
                encoder_out = semantics_enc_out
            elif self.hparams.version == "global" or self.hparams.version == "local":
                audio_enc_out = self.modules.audio_encoder(wavs)
                if self.hparams.downsampling:
                    audio_enc_out = self.modules.conv1(audio_enc_out)
                    audio_enc_out = self.hparams.conv_activation(audio_enc_out)
                    audio_enc_out = self.hparams.dropout(audio_enc_out)
                    audio_enc_out = self.modules.conv2(audio_enc_out)
                    audio_enc_out = self.hparams.conv_activation(audio_enc_out)
                    audio_enc_out = self.hparams.dropout(audio_enc_out)
                # encoder_out = self.modules.fusion(torch.cat((semantics_enc_out, audio_enc_out), dim=-2))
                enc_concat = torch.cat((semantics_enc_out, audio_enc_out), dim=-2)
                encoder_out, _ = self.modules.fusion(enc_concat)
            else:
                raise KeyError('hparams attribute "version" should be one of "cascade-nlu", "global" or "local".')
        except:
            with open("./error_with_file.txt", "a") as errors:
                for element_id in batch.id:
                    errors.write(element_id+'\n')
            encoder_out = semantics_enc_out
        decoder_out = self.modules.decoder(encoder_hidden_states=encoder_out, decoder_input_ids=outputs_tokens)

        logprobs = self.hparams.log_softmax(decoder_out)

        hyps = None

        if self.step % self.hparams.debug_print == 0:
            hyps = torch.argmax(logprobs, dim=-1)
            # hyps = []
            # with torch.no_grad():
            #     for enc_out in encoder_out.detach():
            #         # For greedy search wavs_lens is not used
            #         hyps, _ = self.hparams.valid_greedy_search(enc_out.unsqueeze(0), wavs_lens)
            #         hyps.append(hyp[0])

        if stage != sb.Stage.TRAIN:
            with torch.no_grad():
                hyps, _ = self.hparams.valid_greedy_search(encoder_out.detach(), wavs_lens)

        if stage == sb.Stage.VALID:
            
            # FIXME: The logprobs used here should be the ones corresponding to the decoding loop of the searcher.
            loss = self.compute_objectives((logprobs, hyps, wavs_lens), batch, stage)
            # Logging the Valid Loss
            self.hparams.train_logger.writer.add_scalar(tag="Valid Loss", scalar_value=loss.detach().cpu(), 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step}, 
                                                        train_stats={"Valid Loss: ": loss.detach().cpu()})
            target_tokens, target_lens = batch.outputs_tokens_nobos
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
                    pred_file.write(f'{element_id},{self.tokenizer.decode(hyp)}\n')
                    # Keeping track of the last predicted state of each dialogue to use it for the next prediction
                    if not self.hparams.gold_previous_state:
                        if "SpokenWoz" in element_id:
                            # Id in the form /path/to/dialogue_Turn-N
                            dialog_id = element_id.split("/")[-1].split("_")[0]
                        else:
                            # Id in the form /path/to/dialogue/Turn-N
                            dialog_id = element_id.split("/")[-2]
                        json_state = dialogueState_str2dict(self.tokenizer.decode(hyp))
                        state = dialogueState_dict2str(json_state)
                        with open(os.path.join(self.hparams.output_folder, "last_turns", f'{dialog_id}.txt'), "w") as last_turn:
                            last_turn.write(state + "\n")

        return logprobs, hyps, wavs_lens
    
    def compute_objectives(self, predictions, batch, stage):
        """
        Computes and returns the loss.
        """
        logprobs, hyps, wavs_lens = predictions
        batch = batch.to(self.device)
        outputs_tokens, outputs_lens = batch.outputs_tokens_nobos

        # Replacing the speechbrain padding tokens (id 0) with T5 tokenizer's padding id
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
                if not self.hparams.audio_frozen:
                    self.audio_optimizer.step()
                if not self.hparams.semantic_encoder_frozen:
                    self.semantic_optimizer.step()
                self.decoder_optimizer.step()
                self.fusion_optimizer.step()
            
            if not self.hparams.audio_frozen:
                self.audio_optimizer.zero_grad()
            if not self.hparams.semantic_encoder_frozen:
                self.semantic_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.fusion_optimizer.zero_grad()
            self.optimizer_step += 1

            # Update all the learning rates
            if not self.hparams.audio_frozen:
                old_audio_lr, new_audio_lr = self.hparams.lr_annealing_audio(
                    self.audio_optimizer
                )
            if not self.hparams.semantic_encoder_frozen:
                old_semantic_lr, new_semantic_lr = self.hparams.lr_annealing_semantics(
                    self.semantic_optimizer
                )
            self.hparams.lr_annealing_fusion(
                self.fusion_optimizer
            )
            old_decoder_lr, new_decoder_lr = self.hparams.lr_annealing_decoder(
                self.decoder_optimizer
            )

            # Logging the loss and lrs
            self.hparams.train_logger.writer.add_scalar(tag="Loss", scalar_value=loss.detach().cpu(), 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step}, 
                                                        train_stats={"Loss: ": loss.detach().cpu()})
            
            self.hparams.train_logger.writer.add_scalar(tag="Fusion LR", scalar_value=self.fusion_optimizer.param_groups[0]["lr"], 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step},
                                                train_stats={"Fusion LR": self.fusion_optimizer.param_groups[0]["lr"]})
            
            if not self.hparams.audio_frozen:
                self.hparams.train_logger.writer.add_scalar(tag="Audio Encoder LR", scalar_value=self.audio_optimizer.param_groups[0]["lr"], 
                                                            global_step=self.optimizer_step)
                self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step},
                                                        train_stats={"Audio Encoder LR": self.audio_optimizer.param_groups[0]["lr"]})
            if not self.hparams.semantic_encoder_frozen:
                self.hparams.train_logger.writer.add_scalar(tag="Semantic Encoder LR", scalar_value=self.semantic_optimizer.param_groups[0]["lr"], 
                                                            global_step=self.optimizer_step)
                self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step},
                                                        train_stats={"Semantic Encoder LR": self.semantic_optimizer.param_groups[0]["lr"]})
            self.hparams.train_logger.writer.add_scalar(tag="Decoder LR", scalar_value=self.decoder_optimizer.param_groups[0]["lr"], 
                                                        global_step=self.optimizer_step)
            self.hparams.text_logger.log_stats(stats_meta={"Step": self.optimizer_step},
                                                    train_stats={"Decoder LR": self.decoder_optimizer.param_groups[0]["lr"]})

        # Log the predictions and expected outputs for debug
        if debug_step:
            
            outputs_tokens, _ = batch.outputs_tokens_nobos
            semantics_tokens, _ = batch.semantics_tokens
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
        Initializing the three different optimizers.
        """
        if not self.hparams.audio_frozen:
            if self.hparams.freeze_feature_extractor:
                audio_params = [param for name, param in self.modules.audio_encoder.named_parameters() 
                                if "model.feature_extractor" in name or "model.feature_projection" in name]
            else:
                audio_params = self.modules.audio_encoder.parameters()
            self.audio_optimizer = self.hparams.audio_opt_class(
                audio_params
            )

        if not self.hparams.semantic_encoder_frozen:
            self.semantic_optimizer = self.hparams.semantic_opt_class(
                self.modules.semantic_encoder.parameters()
            )

        fusion_params = self.modules.fusion.parameters()
        if self.hparams.downsampling:
            fusion_params = itertools.chain.from_iterable([fusion_params, 
                                                        self.modules.conv1.parameters(), 
                                                        self.modules.conv2.parameters()])
        self.fusion_optimizer = self.hparams.fusion_opt_class(
            fusion_params
        )

        self.decoder_optimizer = self.hparams.decoder_opt_class(
            self.modules.decoder.parameters()
        )

    def on_stage_start(self, stage, epoch):
        self.stage = stage
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
    
    def _fit_train(self, train_set, epoch, enable):
        
        profiling_activated = self.hparams.profiling_activated
        path_profiling = self.hparams.train_log
        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        steps_since_ckpt = 0
        from utils.profiler import TorchProfilerContextManager
        with TorchProfilerContextManager(profiling_activated, path_profiling) as prof_context:
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
                colour=self.tqdm_barcolor["train"],
            ) as t:
                for batch in t:
                    if self._optimizer_step_limit_exceeded:
                        logger.info("Train iteration limit exceeded")
                        break
                    self.step += 1
                    steps_since_ckpt += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if profiling_activated:
                        prof_context.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0
        self.valid_step = 0