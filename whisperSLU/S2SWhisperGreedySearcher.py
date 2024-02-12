# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

from speechbrain.decoders.seq2seq import S2SGreedySearcher, S2SBeamSearcher
from speechbrain.decoders.seq2seq import _update_mem
import torch

class S2SWhisperGreedySearch(S2SGreedySearcher):
    """
    This class implements the greedy decoding
    for Whisper neural nets made by OpenAI in
    https://cdn.openai.com/papers/whisper.pdf.

    Arguments
    ---------
    model : HuggingFaceWhisper
        The Whisper model.
    language_token : int
        The language token to be used for the decoder input.
    bos_token : int
        The beginning of sentence token to be used for the decoder input.
    task_token : int
        The task token to be used for the decoder input.
    timestamp_token : int
        The timestamp token to be used for the decoder input.
    max_length : int
        The maximum decoding steps to perform.
        The Whisper model has a maximum length of 448.
    **kwargs
        see S2SBaseSearcher, arguments are directly passed.
    """

    def __init__(
        self,
        model,
        language_token=50259,
        bos_token=50258,
        task_token=50359,
        timestamp_token=50363,
        max_length=448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.decoder_input_tokens = None
        self.language_token = language_token  # default language is english
        self.bos_token = bos_token  # always this value
        self.task_token = task_token  # default task is transcribe
        self.timestamp_token = timestamp_token  # default is notimestamp
        self.max_length = max_length - 3  # 3 tokens are added to the input

    def set_language_token(self, language_token):
        """set the language token to be used for the decoder input."""
        self.language_token = language_token

    def set_bos_token(self, bos_token):
        """set the bos token to be used for the decoder input."""
        self.bos_token = bos_token

    def set_task_token(self, task_token):
        """set the task token to be used for the decoder input."""
        self.task_token = task_token

    def set_timestamp_token(self, timestamp_token):
        """set the timestamp token to be used for the decoder input."""
        self.timestamp_token = timestamp_token
        # need to reset bos_index too as timestamp_token is the first
        # inp_token and need to be the first so that the first input gave
        # to the model is [bos, language, task, timestamp] (order matters).
        self.bos_index = self.timestamp_token

    def set_decoder_input_tokens(self, decoder_input_tokens, prompt=None):
        """decoder_input_tokens are the tokens used as input to the decoder.
        They are directly taken from the tokenizer.prefix_tokens attribute.

        decoder_input_tokens = [bos_token, language_token, task_token, timestamp_token]
        """
        self.set_bos_token(decoder_input_tokens[0])
        self.set_language_token(decoder_input_tokens[1])
        self.set_task_token(decoder_input_tokens[2])
        self.set_timestamp_token(decoder_input_tokens[3])

        self.decoder_input_tokens = prompt if prompt is not None else []
        # bos will be timestamp in our case.
        self.decoder_input_tokens += [
            self.bos_token,
            self.language_token,
            self.task_token,
        ]

    def reset_mem(self, batch_size, device):
        """This method set the first tokens to be decoder_input_tokens during search."""
        return torch.tensor([self.decoder_input_tokens] * batch_size).to(device)

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        memory = _update_mem(inp_tokens, memory)

        # WARNING: the max_decode_ratio need to be under 449 because
        #  of positinal encoding
        dec_out, attn = self.model.forward_decoder(enc_states, memory)
        log_probs = self.softmax(dec_out[:, -1])

        return log_probs, memory, attn

    def change_max_decoding_length(self, min_decode_steps, max_decode_steps):
        """set the minimum/maximum length the decoder can take."""
        return (
            int(self.min_decode_ratio * self.max_length),
            int(self.max_decode_ratio * self.max_length),
        )

class S2SWhisperBeamSearch(S2SBeamSearcher):
    """This class implements the beam search decoding
    for Whisper neural nets made by OpenAI in
    https://cdn.openai.com/papers/whisper.pdf.

    Arguments
    ---------
    module : list with the followings one:
        model : torch.nn.Module
            A whisper model. It should have a decode() method.
        ctc_lin : torch.nn.Module (optional)
            A linear output layer for CTC.
    language_token : int
        The token to use for language.
    bos_token : int
        The token to use for beginning of sentence.
    task_token : int
        The token to use for task.
    timestamp_token : int
        The token to use for timestamp.
    max_length : int
        The maximum decoding steps to perform.
        The Whisper model has a maximum length of 448.
    **kwargs
        Arguments to pass to S2SBeamSearcher
    """

    def __init__(
        self,
        module,
        temperature=1.0,
        temperature_lm=1.0,
        language_token=50259,
        bos_token=50258,
        task_token=50359,
        timestamp_token=50363,
        max_length=447,
        **kwargs,
    ):
        super(S2SWhisperBeamSearch, self).__init__(**kwargs)

        self.model = module[0]
        if len(module) == 2:
            self.ctc_fc = module[1]

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm

        self.decoder_input_tokens = None
        self.language_token = language_token  # default language is english
        self.bos_token = bos_token  # always this value
        self.task_token = task_token  # default task is transcribe
        self.timestamp_token = timestamp_token  # default is notimestamp

        self.max_length = max_length - 3  # -3 for [bos, language, task]

    def set_language_token(self, language_token):
        """set the language token to use for the decoder input."""
        self.language_token = language_token

    def set_bos_token(self, bos_token):
        """set the bos token to use for the decoder input."""
        self.bos_token = bos_token

    def set_task_token(self, task_token):
        """set the task token to use for the decoder input."""
        self.task_token = task_token

    def set_timestamp_token(self, timestamp_token):
        """set the timestamp token to use for the decoder input."""
        self.timestamp_token = timestamp_token
        # need to reset bos_index too as timestamp_token is the first
        # inp_token and need to be the first so that the first input gave
        # to the model is [bos, language, task, timestamp] (order matters).
        self.bos_index = self.timestamp_token

    def change_max_decoding_length(self, min_decode_steps, max_decode_steps):
        """set the minimum/maximum length the decoder can take."""
        return (
            int(self.min_decode_ratio * self.max_length),
            int(self.max_decode_ratio * self.max_length),
        )

    def set_decoder_input_tokens(self, decoder_input_tokens, prompt=None):
        """decoder_input_tokens are the tokens used as input to the decoder.
        They are directly taken from the tokenizer.prefix_tokens attribute.

        decoder_input_tokens = [bos_token, language_token, task_token, timestamp_token]
        """
        self.set_bos_token(decoder_input_tokens[0])
        self.set_language_token(decoder_input_tokens[1])
        self.set_task_token(decoder_input_tokens[2])
        self.set_timestamp_token(decoder_input_tokens[3])

        self.decoder_input_tokens = prompt if prompt is not None else []
        # bos will be timestamp in our case.
        self.decoder_input_tokens += [
            self.bos_token,
            self.language_token,
            self.task_token,
        ]

    def reset_mem(self, batch_size, device):
        """This method set the first tokens to be decoder_input_tokens during search."""
        return torch.tensor([self.decoder_input_tokens] * batch_size).to(device)

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        return None

    def permute_mem(self, memory, index):
        """Permutes the memory."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        """Permutes the memory of the language model."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        memory = _update_mem(inp_tokens, memory)
        dec_out, attn, = self.model.forward_decoder(enc_states, memory)
        log_probs = self.softmax(dec_out[:, -1])
        return log_probs, memory, attn

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the implemented LM module."""
        memory = _update_mem(inp_tokens, memory)
        if not next(self.lm_modules.parameters()).is_cuda:
            self.lm_modules.to(inp_tokens.device)
        logits = self.lm_modules(memory)
        log_probs = self.softmax(logits / self.temperature_lm)
        return log_probs[:, -1, :], memory
