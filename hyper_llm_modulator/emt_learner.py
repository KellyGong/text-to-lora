import argparse
import logging
import math
import os
from copy import deepcopy
from types import MethodType
from math import sqrt
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_config, load_peft_weights, PeftConfig, PeftModel
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import yaml

from hyper_llm_modulator.utils import (
    get_layers,
    get_lora_module_names,
    lora_state_dict_to_tensor_dict,
    get_model_and_tokenizer,
    get_pooling_fn,
    add_full_stop,
    get_target_lora_dirs,
    lora_tensor_dict_to_state_dict,
    get_mean_lora,
    get_std_lora,
)



def get_module_names(model, target_modules, layer_indices):
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in model.state_dict():
        if not k.startswith("model.layers."):
            continue
        layer_idx = int(k.split("model.layers.")[-1].split(".")[0])
        if layer_idx in layer_indices:
            for target_module in target_modules:
                if target_module in k:
                    module_names[target_module][layer_idx].append(k)
                    break
    return module_names


def create_emt_learner(
    args, peft_config, device, model, layer_indices
) -> nn.Module:
    """
    Create an EMT learner module for the given model and PEFT configuration.
    """

    module_names = get_module_names(model, args.target_modules, layer_indices)

    emt_learner = EMTLearner(model=model,
                             peft_config=peft_config,
                             module_names=module_names,
                             training_task=args.training_task
                             ).to(device)
    return emt_learner


def zero_lora_param_dict(target_modules, n_layers, rank, in_features, out_features):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(mean=0, std=0.01, size=(n_layers, in_features[m], rank)), requires_grad=True
                    )
                    for m in target_modules
                }
            ),
            "B": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, rank, out_features[m]), requires_grad=True
                    )
                    for m in target_modules
                }
            ),
            "D": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(mean=0, std=1, size=(n_layers, rank)), requires_grad=True
                        # torch.zeros(n_layers, rank), requires_grad=True
                    )
                    for m in target_modules
                }
            )
        }
    )


def get_in_out_features(
    model: PeftModel,
    target_modules: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    in_features = dict()
    out_features = dict()
    for k, module in model.named_modules():
        for target_module in target_modules:
            if target_module in k:        
                if target_module not in in_features:
                    in_features[target_module] = module.in_features
                    out_features[target_module] = module.out_features
                else:
                    # assumes each module has the same input and output features
                    assert in_features[target_module] == module.in_features
                    assert out_features[target_module] == module.out_features

    return in_features, out_features


class EMTLearner(nn.Module):
    """
    EMT Learner module for training with an efficient multi-task lora finetuning.
    """
    def __init__(
        self,
        model: PeftModel,
        peft_config: PeftConfig,
        module_names: list[str],
        training_task: Literal["sft"] = "sft",
        dtype: torch.dtype = torch.float32
    ):

        super().__init__()
        self.model_config = model_config = model.config
        self.peft_config = peft_config
        self.module_names = module_names
        self.scaling = peft_config.lora_alpha / peft_config.r
        self.target_modules = peft_config.target_modules
        self.training_task = training_task

        self.max_num_layers = model_config.num_hidden_layers
        self.device = device = model.device
        self.dtype = dtype

        self.module_to_int = {
            module_name: i for i, module_name in enumerate(module_names)
        }

        self.in_features, self.out_features = get_in_out_features(model, self.target_modules)

        self.SVD_offset = zero_lora_param_dict(
            self.target_modules,
            self.max_num_layers,
            peft_config.r,
            self.in_features,
            self.out_features
        )
    
    # def get_delta_weights(
    #     self,
    #     layer_indices: torch.Tensor,
    #     layer_type: str,
    #     encoded_task_emb: torch.Tensor = None,
    #     factorized: Optional[bool] = None,
    # ) -> torch.Tensor:
    #     """
    #     Get the delta weights for the specified layer type and indices.
    #     """
    #     if factorized is None:

        
    


def save_emt_checkpoint(save_dir, emt_learner, curstep):
    save_path = f"{save_dir}/checkpoints/it_{curstep}/emt.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(emt_learner.state_dict(), save_path)
    return save_path


def load_emt_checkpoint(checkpoint_path, device):
    base_dir = os.path.dirname(checkpoint_path)
    if "checkpoint" in base_dir:
        base_dir = base_dir.split("checkpoint")[0]

    args = argparse.Namespace(
        **yaml.safe_load(open(f"{base_dir}/args.yaml", "r"))
    )
    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{base_dir}/adapter_config.json")
    )
    peft_type = peft_config.peft_type.lower()
    state_dict = torch.load(os.path.join(checkpoint_path, "emt.pt"), map_location=device)

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )
    # train to output delta_w for all layers
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )

    emt_learner = create_emt_learner(
        args, peft_type, device, model, layer_indices
    )

    info = emt_learner.load_state_dict(state_dict, strict=False)
    print(f"Loaded emt module state dict: {info}")
    emt_learner.eval().to(device)
    return (
        args,
        emt_learner,
        model,
        tokenizer
    )
