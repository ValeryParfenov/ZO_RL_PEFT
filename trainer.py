# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

from collections import defaultdict
import inspect
import json
import math
import os
import re
import random
import shutil
import sys
import scipy.stats as sps
from scipy.stats import ortho_group
import time
from functools import partial
from pathlib import Path
from typing import Dict, Union, Any
from typing import TYPE_CHECKING, Optional, Iterable
from transformers.optimization import (
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.func import functional_call, jvp
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
    is_fairscale_available,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# from torch.optim.optimizer import StateDict, params_t
import wandb
# from clearml import Task
from training_utils import * 

from gradient_pruning.pruning_utils import (
    fast_random_mask_like,
    estimate_pretrained_model_magnitude_pruning_threshold,
    compute_named_parameters_to_sparsity,
)
from metrics import f1
from utils import OPT_PERTURBATION_LEVEL_TO_REGEX
from optimizers import * 

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Global storage for metrics history in trainer
_trainer_metrics_history = {}
# Global function for logging metrics from optimizers
_optimizer_log_func = None

def set_optimizer_log_func(log_func):
    """Set the logging function to be used by optimizers"""
    global _optimizer_log_func
    _optimizer_log_func = log_func

def _update_trainer_metrics_history(metrics_dict, step, log_dir):
    """Update the history of metrics for trainer"""
    global _trainer_metrics_history
    # Use log_dir as key to separate different runs
    if log_dir not in _trainer_metrics_history:
        _trainer_metrics_history[log_dir] = {}
    
    history = _trainer_metrics_history[log_dir]
    for metric_name, metric_value in metrics_dict.items():
        if metric_name not in history:
            history[metric_name] = []
        if step is not None:
            history[metric_name].append((step, metric_value))
        else:
            history[metric_name].append((len(history[metric_name]), metric_value))

def _create_trainer_plot(metric_name, history, output_path):
    """Create and save a plot for a metric"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        if not history:
            return
        
        steps, values = zip(*history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} over Training Steps', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot (overwrite existing)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available, skipping plot creation")
    except Exception as e:
        logger.warning(f"Failed to create plot for {metric_name}: {e}")

# Helper function for file logging
def _convert_to_python_types(value):
    """Convert PyTorch tensors and numpy arrays to Python native types"""
    import torch
    import numpy as np
    if torch.is_tensor(value):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

def _log_to_file_if_enabled(metrics_dict, step=None, log_dir=None):
    """Write metrics to local .txt files and update plots if file logging is enabled"""
    if log_dir is None:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    import json
    from datetime import datetime
    
    # Convert all values to Python native types
    metrics_dict = {k: _convert_to_python_types(v) for k, v in metrics_dict.items()}
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update metrics history
    _update_trainer_metrics_history(metrics_dict, step, log_dir)
    
    # Write to a main log file
    main_log_file = os.path.join(log_dir, "training_log.txt")
    with open(main_log_file, "a", encoding="utf-8") as f:
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "metrics": metrics_dict
        }
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    # Write individual metric files and create/update plots
    history = _trainer_metrics_history.get(log_dir, {})
    for metric_name, metric_value in metrics_dict.items():
        # Write metric file
        metric_file = os.path.join(log_dir, f"{metric_name}.txt")
        with open(metric_file, "a", encoding="utf-8") as f:
            if step is not None:
                f.write(f"{step}\t{metric_value}\n")
            else:
                f.write(f"{timestamp}\t{metric_value}\n")
        
        # Create/update plot for this metric
        if metric_name in history:
            plot_path = os.path.join(log_dir, f"{metric_name}.png")
            _create_trainer_plot(metric_name, history[metric_name], plot_path)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    # ZO-Bench added: new parameters to our traininer
    def __init__(self, evaluate_func, dev_samples, eval_samples, perturb_module_regex, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.evaluate_func = evaluate_func
        self.dev_samples = dev_samples
        self.eval_samples = eval_samples
        self.perturb_module_regex = perturb_module_regex
        self.gradient_sparsity = None # FIXME: is it ok?

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to also log to files"""
        # Call parent log method
        super().log(logs)
        # Also log to files if enabled
        if hasattr(self.args, 'log_to_file') and self.args.log_to_file:
            log_dir = getattr(self.args, 'log_run_dir', None) or getattr(self.args, 'log_dir', None) or "logs"
            # Use global_step from state
            step = getattr(self.state, 'global_step', None)
            _log_to_file_if_enabled(logs, step=step, log_dir=log_dir)

    def create_optimizer_and_scheduler(self, num_training_steps):
        # self.optimizer = ""
        # self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer)
        pass

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()  # ----newly-added

        # clearml_task = Task.current_task()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type in ["opt", "llama", "mistral"]:
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2", "llama", "mistral"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # overload the optimizer here
        defaults = {
            'lr': args.learning_rate,
            'momentum': args.momentum,
            'eps': args.zo_eps,
        }

        params = [p for p in self.model.parameters() if p.requires_grad]

        if args.trainer == "zo_adam":
            self.optimizer = ZO_Adam(params, lr=args.learning_rate, eps=args.zo_eps, momentum=args.momentum, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "zo_sgd":
            self.optimizer = ZO_SGD(params, lr=args.learning_rate, eps=args.zo_eps, momentum=args.momentum, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "zo_signsgd":
            self.optimizer = ZO_SignSGD(params, lr=args.learning_rate, eps=args.zo_eps, momentum=args.momentum, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "zo_conserv":
            self.optimizer = ZO_Conserv(params, lr=args.learning_rate, eps=args.zo_eps, momentum=args.momentum, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "jaguar_signsgd":
            self.optimizer = Jaguar_SignSGD(params, lr=args.learning_rate, eps=args.zo_eps, beta=args.zo_beta, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "zo_muon":
            self.optimizer = ZO_MUON(params, lr=args.learning_rate, eps=args.zo_eps, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "zo_sampling_muon":
            self.optimizer = ZO_SamplingMUON(params, lr=args.learning_rate, eps=args.zo_eps, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "jaguar_muon":
            self.optimizer = Jaguar_MUON(params, lr=args.learning_rate, eps=args.zo_eps, beta=args.zo_beta, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type)
        elif args.trainer == "sparse_jaguar_signsgd":
            self.optimizer = Sparse_Jaguar_SignSGD(params, lr=args.learning_rate, eps=args.zo_eps, beta=args.zo_beta, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type, params_ratio=args.params_ratio)
        elif args.trainer == "sparse_jaguar_muon":
            self.optimizer = Sparse_Jaguar_MUON(params, lr=args.learning_rate, eps=args.zo_eps, beta=args.zo_beta, perturbation_mode=args.perturbation_mode, tensor_sampling_type=args.tensor_sampling_type, matrix_sampling_type=args.matrix_sampling_type, params_ratio=args.params_ratio)
        elif args.trainer == "zo_rl_jaguar":
            if args.use_grad_first:
                logger.info("Computing initial gradients for mu0 initialization")
                first_batch = next(iter(train_dataloader))
                first_batch = self._prepare_inputs(first_batch)
                self.model.train()
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(self.model, first_batch)
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
            self.optimizer = ZO_RL(
                params=params, 
                lr=args.learning_rate, 
                eps=args.zo_eps, 
                beta=args.zo_beta,
                perturbation_mode=args.perturbation_mode,
                tensor_sampling_type=args.tensor_sampling_type,
                matrix_sampling_type=args.matrix_sampling_type,
                params_ratio=args.params_ratio,
                k=args.k_value, 
                variance=args.variance, 
                lr_mu=args.lr_mu, 
                use_grad_first=args.use_grad_first
            )
            # Set logging function for optimizer if file logging is enabled
            if hasattr(args, 'log_to_file') and args.log_to_file:
                log_dir = getattr(args, 'log_run_dir', None) or getattr(args, 'log_dir', None) or "logs"
                def optimizer_log_func(metrics_dict, step):
                    _log_to_file_if_enabled(metrics_dict, step=step, log_dir=log_dir)
                set_optimizer_log_func(optimizer_log_func            )
            # Set logging function for optimizer if file logging is enabled
            if hasattr(args, 'log_to_file') and args.log_to_file:
                log_dir = getattr(args, 'log_run_dir', None) or getattr(args, 'log_dir', None) or "logs"
                def optimizer_log_func(metrics_dict, step):
                    _log_to_file_if_enabled(metrics_dict, step=step, log_dir=log_dir)
                set_optimizer_log_func(optimizer_log_func)
        elif args.trainer == "zo_rl_sgd":
            if args.use_grad_first:
                logger.info("Computing initial gradients for mu0 initialization")
                first_batch = next(iter(train_dataloader))
                first_batch = self._prepare_inputs(first_batch)
                self.model.train()
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(self.model, first_batch)
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
            self.optimizer = ZO_RL_SGD(
                params=params, 
                lr=args.learning_rate, 
                eps=args.zo_eps, 
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                tensor_sampling_type=args.tensor_sampling_type,
                matrix_sampling_type=args.matrix_sampling_type,
                perturbation_mode=args.perturbation_mode,
                k=args.k_value, 
                variance=args.variance, 
                lr_mu=args.lr_mu, 
                use_grad_first=args.use_grad_first
            )
            # Set logging function for optimizer if file logging is enabled
            if hasattr(args, 'log_to_file') and args.log_to_file:
                log_dir = getattr(args, 'log_run_dir', None) or getattr(args, 'log_dir', None) or "logs"
                def optimizer_log_func(metrics_dict, step):
                    _log_to_file_if_enabled(metrics_dict, step=step, log_dir=log_dir)
                set_optimizer_log_func(optimizer_log_func)
        elif args.trainer == "zo_adamm":
            self.optimizer = ZO_AdaMM(
                params=params, 
                lr=args.learning_rate, 
                eps=args.zo_eps, 
                betas=(args.beta1, args.beta2), 
                perturbation_mode=args.perturbation_mode, 
                tensor_sampling_type=args.tensor_sampling_type, 
                matrix_sampling_type=args.matrix_sampling_type
            )
        elif args.trainer == "zo_rl_adamm":
            self.optimizer = ZO_RL_AdaMM(
                params=params, 
                lr=args.learning_rate, 
                eps=args.zo_eps, 
                betas=(args.beta1, args.beta2), 
                perturbation_mode=args.perturbation_mode, 
                tensor_sampling_type=args.tensor_sampling_type, 
                matrix_sampling_type=args.matrix_sampling_type,
                variance=args.variance,
                lr_mu=args.lr_mu,
                use_grad_first=args.use_grad_first,
                k=args.k_value
            )
            # Set logging function for optimizer if file logging is enabled
            if hasattr(args, 'log_to_file') and args.log_to_file:
                log_dir = getattr(args, 'log_run_dir', None) or getattr(args, 'log_dir', None) or "logs"
                def optimizer_log_func(metrics_dict, step):
                    _log_to_file_if_enabled(metrics_dict, step=step, log_dir=log_dir)
                set_optimizer_log_func(optimizer_log_func)
        elif args.trainer == "zoar":
            self.optimizer = ZoAR(
                params=params,
                lr=args.learning_rate,
                eps=args.zo_eps,
                weight_decay=args.weight_decay,
                betas=(args.beta1, args.beta2),
                num_queries=args.k_value,
                history=args.zoar_history,
                baseline=getattr(args, "zoar_baseline", "history"),
            )
        elif args.trainer == "loren":
            self.optimizer = LOREN(
                params=params,
                lr=args.learning_rate,
                eps=args.zo_eps,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                num_queries=args.k_value,
                damping=args.loren_damping,
                lr_cov=args.loren_lr_cov,
            )
        else:
            # assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam": # FIXME: what to do with this? 
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            else: 
                raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented")

        self.lr_scheduler = get_scheduler(optimizer=self.optimizer, scheduler_type=args.scheduler, num_training_steps=args.num_training_steps,
                                        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio)
        # self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=args.num_training_steps)
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # Main training loop
        total_steps = 0
        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_sparsity = None  # None, float, or dict

        if self.args.sparse_gradient_group == "layer" or self.args.gradient_sparsity is None:
            self.gradient_sparsity = self.args.gradient_sparsity
            print(f"### layer-wise gradient sparsity = {self.gradient_sparsity}")
        elif self.args.sparse_gradient_group == "global":
            threshold = estimate_pretrained_model_magnitude_pruning_threshold(model, self.args.gradient_sparsity)
            self.gradient_sparsity = compute_named_parameters_to_sparsity(model, threshold)
            print(f"### global gradient sparsity, weight magnitude threshold = {threshold}")

        for epoch in range(epochs_trained, num_train_epochs):
            print(f"-------------------------- Training Epoch {epoch} --------------------------")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # Start one epoch training
            for step, inputs in enumerate(epoch_iterator):

                if total_steps % self.args.sparse_gradient_resample_steps == 0:
                    self.sparse_grad_random_seed = np.random.randint(1000000000)

                total_steps += 1

                # torch.cuda.synchronize()
                step_start_time = time.time()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                diag_every = getattr(args, "ldsd_diag_every", 0)
                if diag_every and args.trainer == "zo_rl_sgd" and self.state.global_step % diag_every == 0:
                    self.optimizer.diag_grads = self._true_grads(model, inputs)
                closure = self.create_closure(model, inputs)
                tr_loss_step = self.optimizer.step(closure)
                if getattr(self.optimizer, "last_diag", None) is not None:
                    self._write_ldsd_diag(self.optimizer.last_diag, tr_loss_step)
                    self.optimizer.last_diag = None
                self.lr_scheduler.step()
                # print(f"Step {total_steps}, LR: {self.optimizer.param_groups[0]['lr']}")

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:
                    print(
                        f"=========================> Evaluating at step {total_steps + 1}... <=========================")
                    val_metrics = self.evaluate_func([], self.dev_samples)
                    test_metrics = self.evaluate_func([], self.eval_samples)
                    if "accuracy" in test_metrics:
                        log_dict = {"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]}
                        self.log(log_dict)
                        if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
                            wandb.log(log_dict)
                        if hasattr(self.args, 'log_to_file') and self.args.log_to_file:
                            log_dir = getattr(self.args, 'log_run_dir', None) or getattr(self.args, 'log_dir', None) or "logs"
                            _log_to_file_if_enabled(log_dict, step=total_steps + 1, log_dir=log_dir)
                        # if clearml_task:
                        #     clearml_task.get_logger().report_scalar(
                        #         "Accuracy", "test", test_metrics["accuracy"], total_steps + 1)
                        #     clearml_task.get_logger().report_scalar(
                        #         "Accuracy", "val", val_metrics["accuracy"], total_steps + 1)
                    else:
                        keys = list(test_metrics.keys())
                        log_dict = {}
                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]
                            log_dict['val_' + k] = val_metrics[k]
                        self.log(log_dict)
                        if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
                            wandb.log(log_dict)
                        # if clearml_task:
                        #     for k, v in log_dict.items():
                        #         clearml_task.get_logger().report_scalar(
                        #             k.split('_')[0], k, v, total_steps + 1)

                max_memory_allocated = 0
                for device_id in range(torch.cuda.device_count()):
                    # this is not accurate since max memory does not happen simultaneously across all devices
                    max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                # Convert loss to Python float
                loss_value = tr_loss_step.item() if torch.is_tensor(tr_loss_step) else float(tr_loss_step)
                log_dict = {"loss": loss_value,
                           "peak_mem": max_memory_allocated / 1024 ** 3,
                           "step_consumption": train_step_duration * 1000}
                self.log(log_dict)
                if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
                    wandb.log(log_dict)
                if hasattr(self.args, 'log_to_file') and self.args.log_to_file:
                    log_dir = getattr(self.args, 'log_run_dir', None) or getattr(self.args, 'log_dir', None) or "logs"
                    _log_to_file_if_enabled(log_dict, step=total_steps, log_dir=log_dir)
                # if clearml_task:
                #     clearml_task.get_logger().report_scalar(
                #         "Memory", "peak_mem", max_memory_allocated / 1024 ** 3, total_steps)
                #     clearml_task.get_logger().report_scalar(
                #         "Time", "step_consumption", train_step_duration * 1000, total_steps)

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
            wandb.log(metrics)
        self.log(metrics)
        if hasattr(self.args, 'log_to_file') and self.args.log_to_file:
            log_dir = getattr(self.args, 'log_run_dir', None) or getattr(self.args, 'log_dir', None) or "logs"
            _log_to_file_if_enabled(metrics, step=self.state.global_step, log_dir=log_dir)
        # if clearml_task:
        #     for key, value in metrics.items():
        #         clearml_task.get_logger().report_scalar(
        #             "Training", key, value, self.state.global_step)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _true_grads(self, model, inputs):
        """Minibatch gradient at x_t: same batch, loss and eval mode as the ZO closure."""
        model.eval()
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        with torch.enable_grad():
            batch = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, batch)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, grads)]

    def _write_ldsd_diag(self, diag, loss):
        """Append one diagnostics record to <output_dir>/ldsd_diag.jsonl and log a summary."""
        record = {"step": self.state.global_step, "loss": float(loss), **diag}
        with open(os.path.join(self.args.output_dir, "ldsd_diag.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("ldsd_diag step=%d mu_ratio=%.3e cos_mu=%.3e cos_sel=%.3e C_mean=%.3e C_iso=%.3e",
                    record["step"], diag["mu_ratio"], diag["cos_mu"], diag["cos_sel"],
                    diag["C_mean"], diag["C_iso"])

    def create_closure(self, model, inputs):
        '''
        Creates closure for optimizer step.
        '''
        def closure(): return self.zo_forward(model, inputs)
        
        return closure

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # Sparse gradient
        # FIXME: do we need it? 
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            if grad_sparsity is not None:
                param.grad[fast_random_mask_like(param.grad, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        return loss.detach()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
