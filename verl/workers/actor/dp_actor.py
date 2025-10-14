# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_with_entropy, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.get("entropy_from_logits_with_chunking", False):
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            max_prob_log_probs: # (bs, response_len) - log prob of most probable token
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    
                    # Compute log prob of the most probable token in vocabulary
                    max_prob_logits = torch.max(logits_rmpad, dim=-1)[0]  # (total_nnz,)
                    logsumexp_logits = torch.logsumexp(logits_rmpad, dim=-1)  # (total_nnz,)
                    max_prob_log_probs = max_prob_logits - logsumexp_logits  # (total_nnz,)

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.get("entropy_checkpointing", False):
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    # gather max_prob_log_probs
                    max_prob_log_probs = gather_outpus_and_unpad(
                        max_prob_log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                # pad max_prob_log_probs
                full_max_prob_log_probs = pad_input(
                    hidden_states=max_prob_log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                max_prob_log_probs = full_max_prob_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    # Compute log prob of the most probable token in vocabulary
                    max_prob_logits = torch.max(logits, dim=-1)[0]  # (bsz, response_length)
                    logsumexp_logits = torch.logsumexp(logits, dim=-1)  # (bsz, response_length)
                    max_prob_log_probs = max_prob_logits - logsumexp_logits  # (bsz, response_length)
                    
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)


        return entropy, log_probs, max_prob_log_probs

    def _optimizer_step(self):
        assert self.config.get("grad_clip") is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.get("grad_clip"))
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.get("grad_clip"))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.get("grad_clip"))

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy (bool): whether to calculate entropy

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (log_probs, entropys, max_prob_log_probs)
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        max_prob_log_probs_lst = []  # collect max_prob_log_probs
        
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, max_prob_log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            max_prob_log_probs_lst.append(max_prob_log_probs)  # collect max_prob_log_probs
            if calculate_entropy and entropy is not None:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        max_prob_log_probs = torch.concat(max_prob_log_probs_lst, dim=0)  # concat max_prob_log_probs
        
        entropys = None
        if calculate_entropy and entropy_lst:
            entropys = torch.concat(entropy_lst, dim=0)
            
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            max_prob_log_probs = max_prob_log_probs[revert_indices]  # handle max_prob_log_probs
            if calculate_entropy and entropys is not None:
                entropys = entropys[revert_indices]

        return log_probs, entropys, max_prob_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "max_prob_log_probs", "entropys"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.get("use_kl_loss", False):
            select_keys.append("ref_log_prob")

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.get("ppo_mini_batch_size")
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.get("ppo_mini_batch_size"))

        metrics = {}
        all_clip_positions = []  # collect clip position information
        for epoch in range(self.config.get("ppo_epochs", 1)):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.get("use_dynamic_bsz", False):
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.get("ppo_max_token_len_per_gpu", 1024) * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.get("ppo_mini_batch_size") // self.config.get("ppo_micro_batch_size_per_gpu")
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.get("ppo_micro_batch_size_per_gpu")
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.get("use_dynamic_bsz", False):
                    max_token_len = self.config.get("ppo_max_token_len_per_gpu", 1024) * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.get("ppo_mini_batch_size") // self.config.get("ppo_micro_batch_size_per_gpu")
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.get("ppo_micro_batch_size_per_gpu"))

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.get("clip_ratio", 0.2)
                    clip_ratio_low = self.config.get("clip_ratio_low") if self.config.get("clip_ratio_low") is not None else clip_ratio
                    clip_ratio_high = self.config.get("clip_ratio_high") if self.config.get("clip_ratio_high") is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    
                    # Check if there is per-prompt clip_ratio_high
                    if "per_prompt_clip_ratio_high" in data:
                        # Use per-prompt clip_ratio_high
                        per_prompt_clip_ratio_high = torch.tensor(data["per_prompt_clip_ratio_high"], 
                                                                 device=advantages.device, 
                                                                 dtype=advantages.dtype)
                        # Expand to same shape as advantages
                        clip_ratio_high = per_prompt_clip_ratio_high.unsqueeze(-1).expand_as(advantages)
                    else:
                        # Use global clip_ratio_high
                        clip_ratio_high = clip_ratio_high
                    
                    entropy_coeff = self.config.get("entropy_coeff", 0.0)
                    loss_agg_mode = self.config.get("loss_agg_mode", "vanilla")

                    # Based on debug output, config structure is self.config['policy_loss']['loss_mode']
                    loss_mode = self.config.get("policy_loss", {}).get("loss_mode", "vanilla")
                    
                    # all return: (bsz, response_length)
                    # only compute entropy in entropy_control mode
                    if loss_mode == "entropy_control":
                        if "entropys" in data:
                            entropy = data["entropys"]
                            # Re-compute log_prob (without computing entropy)
                            _, log_prob, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=False)
                        else:
                            # If entropys is not passed in, re-compute
                            calculate_entropy = False
                            if entropy_coeff != 0:
                                calculate_entropy = True
                            entropy, log_prob, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    else:
                        # In vanilla mode, decide whether to compute entropy based on entropy_coeff
                        calculate_entropy = False
                        if entropy_coeff != 0:
                            calculate_entropy = True
                        entropy, log_prob, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    if loss_mode == "entropy_control" and entropy is not None:
                        token_weight_min = self.config.get("policy_loss", {}).get("token_weight_min", 0.95)
                        token_weight_max = self.config.get("policy_loss", {}).get("token_weight_max", 1.05)
                        linear = self.config.get("policy_loss", {}).get("linear", True)  # get linear parameter
                        
                        result = compute_policy_loss_with_entropy(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            entropys=entropy,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            token_weight_min=token_weight_min,
                            token_weight_max=token_weight_max,
                            linear=linear,  # linear parameter
                        )
                        # Now returns 9 values, directly unpack
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, pg_clipfrac_low, clipped_by_high, clipped_by_low, clip_stats = result
                    elif loss_mode == "vanilla":
                        # Use standard policy loss function
                        print(f"Entering vanilla branch")
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, pg_clipfrac_low, clipped_by_high, clipped_by_low, clip_stats = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        # Use other registered policy loss functions
                        print(f"Entering other branch, loss_mode: {loss_mode}")
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        result = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config)
                        
                        # Handle different policy_loss_fn return values
                        if len(result) == 4:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
                            # For functions returning 4 values, set default values
                            pg_clipfrac_high = torch.tensor(0.0, device=pg_clipfrac.device)
                            pg_clipfrac_low = torch.tensor(0.0, device=pg_clipfrac.device)
                            clipped_by_high = torch.zeros_like(advantages, dtype=torch.bool)
                            clipped_by_low = torch.zeros_like(advantages, dtype=torch.bool)
                            # Create clip_stats for other policy_loss_fn
                            clip_stats = {
                                "clipped_by_high": clipped_by_high,
                                "clipped_by_low": clipped_by_low,
                                "total_clipped": clipped_by_high | clipped_by_low,
                                "clip_high_count": clipped_by_high.sum().item(),
                                "clip_low_count": clipped_by_low.sum().item(),
                                "total_clipped_count": (clipped_by_high | clipped_by_low).sum().item(),
                            }
                        elif len(result) == 6:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, pg_clipfrac_low = result
                            clipped_by_high = torch.zeros_like(advantages, dtype=torch.bool)
                            clipped_by_low = torch.zeros_like(advantages, dtype=torch.bool)
                            # Create clip_stats for other policy_loss_fn
                            clip_stats = {
                                "clipped_by_high": clipped_by_high,
                                "clipped_by_low": clipped_by_low,
                                "total_clipped": clipped_by_high | clipped_by_low,
                                "clip_high_count": clipped_by_high.sum().item(),
                                "clip_low_count": clipped_by_low.sum().item(),
                                "total_clipped_count": (clipped_by_high | clipped_by_low).sum().item(),
                            }
                        elif len(result) == 8:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, pg_clipfrac_low, clipped_by_high, clipped_by_low = result
                            # Create clip_stats for other policy_loss_fn
                            clip_stats = {
                                "clipped_by_high": clipped_by_high,
                                "clipped_by_low": clipped_by_low,
                                "total_clipped": clipped_by_high | clipped_by_low,
                                "clip_high_count": clipped_by_high.sum().item(),
                                "clip_low_count": clipped_by_low.sum().item(),
                                "total_clipped_count": (clipped_by_high | clipped_by_low).sum().item(),
                            }
                        elif len(result) == 9:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, pg_clipfrac_low, clipped_by_high, clipped_by_low, clip_stats = result
                        else:
                            raise ValueError(f"Unexpected number of return values from policy_loss_fn: {len(result)}")

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.get("use_kl_loss", False):
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.get("kl_loss_type"))
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.get("kl_loss_coef")
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.get("kl_loss_coef")



                    if self.config.get("use_dynamic_bsz", False):
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.get("ppo_mini_batch_size"))
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    # compute probability ratio analysis for clip positions
                    if "max_prob_log_probs" in data:
                        max_prob_log_probs = data["max_prob_log_probs"]
                        
                        # Check dimension matching
                        if old_log_prob.shape != max_prob_log_probs.shape:
                            print(f"[Warning] Shape mismatch: old_log_prob {old_log_prob.shape} vs max_prob_log_probs {max_prob_log_probs.shape}")
                            # Try broadcasting or truncating to same shape
                            min_seq_len = min(old_log_prob.shape[-1], max_prob_log_probs.shape[-1])
                            old_log_prob = old_log_prob[..., :min_seq_len]
                            max_prob_log_probs = max_prob_log_probs[..., :min_seq_len]
                        
                        # Convert log probabilities to actual probabilities
                        old_probs = torch.exp(old_log_prob)  # Convert to actual probability
                        max_probs = torch.exp(max_prob_log_probs)  # Convert to actual probability
                        
                        # numerical stability check
                        if not torch.isfinite(old_probs).all():
                            print(f"[Warning] Non-finite values in old_probs: {torch.isfinite(old_probs).sum()}/{old_probs.numel()}")
                            old_probs = torch.clamp(old_probs, min=1e-10, max=1.0)
                        
                        if not torch.isfinite(max_probs).all():
                            print(f"[Warning] Non-finite values in max_probs: {torch.isfinite(max_probs).sum()}/{max_probs.numel()}")
                            max_probs = torch.clamp(max_probs, min=1e-10, max=1.0)
                        
                        # Compute probability ratio: old_prob / max_prob
                        prob_ratio = old_probs / (max_probs + 1e-10)  # Add small value to avoid division by zero
                        
                        # Compute clip position statistics
                        if clip_stats["total_clipped_count"] > 0:
                            # Get ratios for clip-high and clip-low positions separately
                            clip_high_ratios = prob_ratio[clip_stats["clipped_by_high"]]
                            clip_low_ratios = prob_ratio[clip_stats["clipped_by_low"]]
                            total_clipped_ratios = prob_ratio[clip_stats["total_clipped"]]
                            
                            # Compute clip-high statistics
                            if clip_stats["clip_high_count"] > 0:
                                micro_batch_metrics.update({
                                    "actor/clip_high_ratio_mean": clip_high_ratios.mean().detach().item(),
                                    "actor/clip_high_ratio_min": clip_high_ratios.min().detach().item(),
                                    "actor/clip_high_ratio_max": clip_high_ratios.max().detach().item(),
                                    "actor/clip_high_old_prob_mean": old_probs[clip_stats["clipped_by_high"]].mean().detach().item(),
                                    "actor/clip_high_max_prob_mean": max_probs[clip_stats["clipped_by_high"]].mean().detach().item(),
                                })
                            
                            # Compute clip-low statistics
                            if clip_stats["clip_low_count"] > 0:
                                micro_batch_metrics.update({
                                    "actor/clip_low_ratio_mean": clip_low_ratios.mean().detach().item(),
                                    "actor/clip_low_ratio_min": clip_low_ratios.min().detach().item(),
                                    "actor/clip_low_ratio_max": clip_low_ratios.max().detach().item(),
                                    "actor/clip_low_old_prob_mean": old_probs[clip_stats["clipped_by_low"]].mean().detach().item(),
                                    "actor/clip_low_max_prob_mean": max_probs[clip_stats["clipped_by_low"]].mean().detach().item(),
                                })
                            
                            # Compute overall statistics
                            micro_batch_metrics.update({
                                "actor/total_clip_ratio_mean": total_clipped_ratios.mean().detach().item(),
                                "actor/total_clip_ratio_min": total_clipped_ratios.min().detach().item(),
                                "actor/total_clip_ratio_max": total_clipped_ratios.max().detach().item(),
                                "actor/clip_high_count": float(clip_stats["clip_high_count"]),
                                "actor/clip_low_count": float(clip_stats["clip_low_count"]),
                                "actor/total_clipped_count": float(clip_stats["total_clipped_count"]),
                            })
                        
                            
                            # Save clip position information to data for ray_trainer use
                            clip_position_info = {
                                "clipped_by_high": clip_stats["clipped_by_high"],
                                "clipped_by_low": clip_stats["clipped_by_low"],
                                "total_clipped": clip_stats["total_clipped"],
                                "prob_ratios": prob_ratio,
                                "old_probs": old_probs,
                                "max_probs": max_probs,
                                "clip_high_count": clip_stats["clip_high_count"],
                                "clip_low_count": clip_stats["clip_low_count"],
                                "total_clipped_count": clip_stats["total_clipped_count"],
                            }
                            
                            # Save clip position information based on data type
                            if isinstance(data, dict):
                                data["clip_positions"] = clip_position_info
                            elif hasattr(data, 'batch'):
                                data.batch["clip_positions"] = clip_position_info
                            
                            all_clip_positions.append(clip_position_info)
                        else:
                            # Even without clip, record statistics
                            clip_position_info = {
                                "clipped_by_high": clip_stats["clipped_by_high"],
                                "clipped_by_low": clip_stats["clipped_by_low"],
                                "total_clipped": clip_stats["total_clipped"],
                                "clip_high_count": clip_stats["clip_high_count"],
                                "clip_low_count": clip_stats["clip_low_count"],
                                "total_clipped_count": clip_stats["total_clipped_count"],
                            }
                            
                            # Save clip position information based on data type
                            if isinstance(data, dict):
                                data["clip_positions"] = clip_position_info
                            elif hasattr(data, 'batch'):
                                data.batch["clip_positions"] = clip_position_info
                            
                            all_clip_positions.append(clip_position_info)
                    
                    # Only add token_weights metrics in entropy_control mode
                    if loss_mode == "entropy_control" and "token_weights_mean" in clip_stats:
                        micro_batch_metrics.update({
                            "actor/token_weights_mean": float(clip_stats["token_weights_mean"]),
                            "actor/token_weights_min": float(clip_stats["token_weights_min"]),
                            "actor/token_weights_max": float(clip_stats["token_weights_max"]),
                        })
                    
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            "actor/pg_clipfrac_high": pg_clipfrac_high.detach().item(),
                            "actor/pg_clipfrac_low": pg_clipfrac_low.detach().item(),
                        }
                    )


                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        
        # return clip position information
        return metrics, all_clip_positions
