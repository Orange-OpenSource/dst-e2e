# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

import torch

class TorchProfilerContextManager:
    '''
    Context class to manage TorchProfiler usage
    '''
    def __init__(self, profiling_activated, path_profiling=None):
        self.profiling_activated = profiling_activated
        self.path_profiling = path_profiling
        self.profiler = None

    def __enter__(self):
        if self.profiling_activated:
            # Setup profiling (see https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
            # wait = number of steps to wait before starting profiling (profiler is disabled)
            # warmup = number of steps during which the profiler starts tracking but discards the results (to reduce profiling overhead)
            # active = number of steps during which the profiler works and records events
            # repeat = number of repetition of the cycle (called "span")
            self.profiler = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=160, warmup=160, active=32, repeat=3),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.path_profiling),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True)
            self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiling_activated:
            self.profiler.stop()
