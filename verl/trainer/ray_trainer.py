# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import random
from verl.trainer.data_loader import create_datasets
from verl.trainer.utils import load_json
import re

import json
import os
import subprocess
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
import yaml
from collections import Counter
from ray.experimental.tqdm_ray import tqdm
# Imports needed to create DataLoaders internally
from torch.utils.data import Dataset, RandomSampler, SequentialSampler,Subset, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from ..utils.dataset import collate_fn  # Import collate_fn
from ..utils.dataset import collate_fn, RLHFDataset  # Import collate_fn and RLHFDataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedTokenizer, ProcessorMixin
import matplotlib.pyplot as plt  # Import matplotlib.pyplot as plt

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


# === 新增：辅助数据集，用于不确定性采样 ===
# URST原代码思路.md 中提到，要对每个样本复制k次
class kRepeatDataset(Dataset):
    """一个包装器，将数据集中的每个样本重复k次。"""
    def __init__(self, original_dataset, k):
        self.original_dataset = original_dataset
        self.k = k
        self.original_len = len(self.original_dataset)

    def __len__(self):
        return self.original_len * self.k

    def __getitem__(self, idx):
        # 无论idx是什么，我们都将其映射回原始索引
        original_idx = idx // self.k
        return self.original_dataset[original_idx]
        
    def get_original_indices(self):
        """返回此数据集中唯一的原始索引"""
        return list(range(self.original_len))

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    #TODO:不传入dataloader只传入dataset
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataset: Dataset,
        val_dataset: Dataset,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        # === NEW: DATALOADER CREATION ===
 
        # Logic moved from data_loader.py to here
        
        # # 1. Get batch sizes from config
        # if config.data.mini_rollout_batch_size is not None:
        #     train_batch_size = config.data.mini_rollout_batch_size
        # else:
        #     train_batch_size = config.data.rollout_batch_size
        
        if config.data.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.data.val_batch_size
 
        # # 2. Create train dataloader sampler
        # if config.data.shuffle:
        #     train_dataloader_generator = torch.Generator()
        #     train_dataloader_generator.manual_seed(config.data.seed)
        #     sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        # else:
        #     sampler = SequentialSampler(data_source=train_dataset)
 
        # # 3. Create StatefulDataLoaders
        # self.train_dataloader = StatefulDataLoader(
        #     dataset=train_dataset,
        #     batch_size=train_batch_size,
        #     sampler=sampler,
        #     num_workers=4,  # Hardcoded in original data_loader.py
        #     collate_fn=collate_fn, # Imported from ..utils.dataset
        #     pin_memory=False,
        #     drop_last=True,
        # )
        
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4, # Hardcoded in original data_loader.py
            collate_fn=collate_fn, # Imported from ..utils.dataset
            pin_memory=False,
            drop_last=False,
        )
        self.full_train_dataset = train_dataset
        # ================================
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        # self.train_dataloader 将在 _ppo_train_on_subset 中被动态创建
        self.train_dataloader = None 
        self.data_iterator = None
        self.training_steps = 0  # 将在 PPO 循环开始前设置
        self.global_step = 0

        # self._val_generation_log_counter = 0
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")
        # # modified
        # if (config.data.rollout_batch_size * config.worker.rollout.n) % config.worker.actor.global_batch_size != 0:  
        #     raise ValueError("Rollout batch size * rollout.n must be divisible by actor global batch size.")
        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # if config.trainer.max_steps is not None:
        #     self.training_steps = config.trainer.max_steps
        # elif config.data.mini_rollout_batch_size is not None:
        #     num_examples = len(self.train_dataloader) * config.data.mini_rollout_batch_size
        #     self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        # else:
        #     self.training_steps = len(self.train_dataloader) * config.trainer.total_epochs

        # config.worker.actor.optim.training_steps = self.training_steps
        # config.worker.critic.optim.training_steps = self.training_steps
        # print(f"Total training steps: {self.training_steps}")

        self.global_continuous_step = 0

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self, round_num: int) -> None:
        """
        按轮次保存，并返回 actor_path。
        """
        # global_step 在 _ppo_train_on_subset 中被更新
        current_step = self.global_step
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = f"round_{round_num}_step_{self.global_step}"

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        # 按轮次保存 checkpoint
        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path, 
            f"round_{round_num}",
            f"global_step_{current_step}"
        )
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        # 更新 checkpointer tracker，指向最新的模型
        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": f"round_{round_num}_step_{current_step}",
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
        return actor_path # 返回 actor 路径，用于下一轮加载

    def _load_checkpoint(self) -> str:
        load_checkpoint_path = None # 初始化
        
        if self.config.trainer.find_last_checkpoint:
            # --- [逻辑 1: 优先尝试恢复最新 SGPO 进度] ---
            print(f"[INFO] 正在搜索最新的 SGPO checkpoint: {self.config.trainer.save_checkpoint_path}")
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path) 
            
            if tracker_info is not None: #
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0) #
                self.best_global_step = tracker_info.get("best_global_step", 0) #
        
        if load_checkpoint_path is None:
            # --- [逻辑 2: 没找到 SGPO 检查点，这是第一次运行] ---
            # SFT 模型已经在 init_workers() 中通过 config 加载了
            print("[INFO] 未找到 SGPO checkpoint。将使用在 `init_workers` 中加载的 SFT/Base 模型开始训练。")
            return None
        
        # --- [逻辑 3: 找到了一个 SGPO Checkpoint，加载它] ---
        print(f"[INFO] 找到了最新的 SGPO checkpoint，正在恢复: {load_checkpoint_path}")
        
        potential_actor_path = os.path.join(load_checkpoint_path, "actor") #
        
        if not os.path.exists(potential_actor_path):
             print(f"[WARNING] 路径 {load_checkpoint_path} 不是一个有效的 SGPO checkpoint (缺少 'actor' 目录)。将使用 SFT/Base 模型。")
             return None

        # --- [这是 SGPO Checkpoint 的加载逻辑] ---
        print("[INFO] 检测到 Trainer checkpoint 结构 (包含 'actor' 子目录)。") #
        actor_path = potential_actor_path #
        
        # 尝试解析 SGPO 的 global_step
        if "global_step_" in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]: #
            try:
                step_str = load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1] #
                self.global_step = int(step_str) #
                print(f"[INFO] 解析到 trainer global_step: {self.global_step}")
            except:
                print(f"[WARNING] 无法从 {load_checkpoint_path} 解析 global_step，重置为 0。") #
                self.global_step = 0 #
        else:
            print("[INFO] Checkpoint 路径不含 'global_step_'，重置 trainer 步数为 0。")
            self.global_step = 0
            
        print(f"[INFO] 正在加载 Actor (SGPO 格式): {actor_path}") #
        self.actor_rollout_ref_wg.load_checkpoint(actor_path) #
        
        # 加载 SGPO 的 Critic
        if self.use_critic: #
            critic_path = os.path.join(load_checkpoint_path, "critic") #
            if os.path.exists(critic_path):
                print(f"[INFO] 正在加载 Critic: {critic_path}") #
                self.critic_wg.load_checkpoint(critic_path) #
            else:
                print(f"[WARNING] 未找到 Critic checkpoint: {critic_path}") #
        
        print("[INFO] 已跳过加载 dataloader.pt 状态。") #
        
        return actor_path #
    
    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        # if not samples:
        #     return

        # self._val_generation_log_counter += 1
        # log_step = self.global_step * 1000 + self._val_generation_log_counter
        self.logger.log_generation(samples, self.global_step)


    def _validate(self) -> dict[str, Any]:
        """多次采样（repeat_times），保证评估的稳定性。 这个多次采样是指：对于同一个输入，模型会生成多个不同的输出（答案），而不是只生成一次。这是因为生成式模型有随机性，每次采样可能结果不同。
        这样做的好处是：

        可以更全面地评估模型的表现，避免偶然性。
        统计多个输出的平均分数，更稳定、更可靠。
        在强化学习和生成任务中，通常会设置 repeat_times，比如每个输入采样 3 次、5 次，然后对这些结果做平均或其他统计分析。"""

        def _to_label_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, torch.Tensor):
                return value.cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]

        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        val_predictions, val_ground_truths = [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            parsed_answers = [self.parse_answer_from_output(text) for text in output_texts]
            batch_ground_truths = _to_label_list(test_batch.non_tensor_batch.get("ground_truth"))
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(batch_ground_truths)
            sample_scores.extend(scores)
            val_predictions.extend(parsed_answers)
            val_ground_truths.extend(batch_ground_truths)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        val_gt_upper = [str(gt).upper() for gt in val_ground_truths]
        val_pred_upper = [str(pred).upper() for pred in val_predictions]
        valid_indices = [
            idx for idx, (gt, pred) in enumerate(zip(val_gt_upper, val_pred_upper))
            if gt in ["YES", "NO"] and pred in ["YES", "NO"]
        ]

        if valid_indices:
            filtered_gts = [val_gt_upper[idx] for idx in valid_indices]
            filtered_preds = [val_pred_upper[idx] for idx in valid_indices]
            val_accuracy = accuracy_score(filtered_gts, filtered_preds)
            val_f1 = f1_score(filtered_gts, filtered_preds, average="macro", labels=["YES", "NO"])
        else:
            val_accuracy = 0.0
            val_f1 = 0.0

        metrics = {
            "val/reward_score": self.val_reward_score,
            "val/accuracy": val_accuracy,
            "val/f1_score": val_f1,
            **val_reward_metrics,
            **val_length_metrics,
        }
        print("Finish validation.")
        return metrics

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens
        _balance_batch 这个函数的作用是：在分布式训练时，把一个 batch 里的数据重新排序和分组，让每个 GPU/进程分到的数据“总 token 数”尽量接近，实现负载均衡。
        具体流程如下：
        统计每个样本的 token 数（即 attention_mask 的和）。
        根据 world_size（GPU/进程数），用分区算法把所有样本分成 world_size 组，每组 token 总数尽量均衡。
        按照分组结果重新排列 batch 数据。
        记录分组后的均衡统计信息到 metrics，方便后续分析。"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        """Make a batch of data for rollout._make_batch_data 
        这个函数的作用是：从训练数据集中动态生成一个用于强化学习训练的 batch（批次），并对 batch 进行采样、奖励计算、过滤和拼接，直到满足 batch 大小要求。"""
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            # 没用
            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # DAPO filter group ,online_filtering = false
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    # === 新增：计算熵 ===
    def _calculate_entropy(self, outputs: list[str]) -> float:
        """根据 k 次采样输出（例如 "YES", "NO"）计算熵。"""
        #outputs是类似 ["YES", "YES", "NO", "YES", "UNKNOWN", ...] 的列表
        if not outputs:
            return 0.0
        # 过滤掉无法解析的答案
        valid_outputs = [o for o in outputs if o != "UNKNOWN"]
        k = len(valid_outputs)
        if k == 0:
            return 0.0 # 如果所有输出都是 "UNKNOWN"，熵为0
            
        counts = Counter(valid_outputs)
        entropy = 0.0
        for count in counts.values():
            p = count / k
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    

    def parse_answer_from_output(self,model_output: str) -> str:
        """
        从模型的完整输出字符串中提取 "YES" 或 "NO" 答案。
    
        
        它搜索 <answer>...</answer> 标签之间的内容，
        并将其标准化为 "YES", "NO", 或 "UNKNOWN"。
        """
        # re.DOTALL (s) 使得 '.' 也能匹配换行符
        # re.IGNORECASE (i) 使得 <answer>yes</answer> 也能被匹配
        match = re.search(r'<answer>(.*?)</answer>', model_output, re.IGNORECASE | re.DOTALL)
        
        if match:
            # 提取括号()中的内容，去除首尾空格，转为大写
            answer = match.group(1).strip().upper()
            
            # 检查 "YES" 或 "NO" 是否在提取的内容中
            # 这可以处理 "answer here YES" 或 "YES" 这样的输出
            if "YES" in answer:
                return "YES"
            if "NO" in answer:
                return "NO"
        
        # 回退机制：如果模型没有生成 <answer> 标签，
        # 而是直接输出了 "YES" 或 "NO"，我们也能处理。
        cleaned_output = model_output.strip().upper()
        if cleaned_output == "YES":
            return "YES"
        if cleaned_output == "NO":
            return "NO"
                
        # 如果两种方式都解析失败
        return "UNKNOWN"


    # === 新增：不确定性采样函数 ===
    def _uncertainty_sampling(self, unselected_dataset: Dataset) -> list[tuple[int, str, float]]:
        """
        对所有未标记数据进行 n 次采样并计算熵。
        返回 (original_index, label, entropy) 列表。
        """
        n = self.config.active_learning.k_samples
        if len(unselected_dataset) == 0:
            print("No unselected data left for sampling.")
            return []

        sampler = SequentialSampler(unselected_dataset)
        rollout_bsz = getattr(self.config.data, "rollout_batch_size", 1)
        dataloader = DataLoader(
            unselected_dataset,
            batch_size=rollout_bsz,
            sampler=sampler,
            num_workers=0,  # TODO: 是否需要调整为 0
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        if hasattr(unselected_dataset, "indices"):
            original_indices = list(unselected_dataset.indices)
        else:
            original_indices = list(range(len(unselected_dataset)))
        current_idx = 0
        
        print(f"Running uncertainty sampling (n={n}) on {len(unselected_dataset)} samples...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        samples_with_entropy = []
        
        # dataloader 每次返回多个不同样本，依赖 n 来获得多次生成
        for batch_id, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader), desc="Uncertainty Sampling"):
            gt_data = batch_dict[self.config.data.answer_key]

            def _flatten_labels(data):
                if isinstance(data, torch.Tensor):
                    return data.reshape(-1).cpu().tolist()
                if isinstance(data, np.ndarray):
                    return data.reshape(-1).tolist()
                if isinstance(data, (list, tuple)):
                    return list(data)
                return [data]

            label_values = _flatten_labels(gt_data)

            test_batch = DataProto.from_single_dict(batch_dict)
            batch_size = len(test_batch)
            batch_original_indices = original_indices[current_idx : current_idx + batch_size]
            current_idx += batch_size

            labels = []
            for idx in range(batch_size):
                value = label_values[idx] if idx < len(label_values) else ""
                labels.append(value if isinstance(value, str) else str(value))

            gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            
            # 设置采样参数
            gen_batch.meta_info = self.config.worker.rollout.val_override_config.copy()
            gen_batch.meta_info["do_sample"] = True
            gen_batch.meta_info["temperature"] = 1.0
            gen_batch.meta_info["n"] = n
            # gemini的建议
            gen_batch.meta_info.pop("seed", None)

            # 这里pad和unpad是为了适应多卡环境
            gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            # pad 的每条样本同样会采样 n 次，因此需要按 pad_size * n 去掉填充样本
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * n)

            output_ids = test_output_gen_batch.batch["responses"]
            raw_model_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            parsed_answers = [self.parse_answer_from_output(raw_output) for raw_output in raw_model_outputs]

            if len(parsed_answers) != batch_size * n:
                print(
                    f"Warning - Expected {batch_size * n} answers but got {len(parsed_answers)} in batch {batch_id}."
                )

            for local_idx in range(batch_size):
                start = local_idx * n
                end = start + n
                sample_answers = parsed_answers[start:end]
                if not sample_answers:
                    continue
                print(f"Debug - Sample parsed_answers: {sample_answers}")
                entropy = self._calculate_entropy(sample_answers)
                print(f"Debug - Sample Calculated entropy: {entropy}")
                unknown_count = sample_answers.count("UNKNOWN")
                original_index = batch_original_indices[local_idx]
                if unknown_count > 0:
                    print(
                        f"Sample index {original_index} had {unknown_count}/{n} UNKNOWN answers during uncertainty sampling."
                    )
                samples_with_entropy.append((original_index, labels[local_idx], entropy))

        self.actor_rollout_ref_wg.release_rollout_engine()
        
        # 按熵降序排序
        samples_with_entropy.sort(key=lambda x: x[2], reverse=True)
        
        # 绘制entropy分布图
        entropies = [ent for _,_, ent in samples_with_entropy]
        plt.figure(figsize=(8, 5))
        plt.hist(entropies, bins=30, color='skyblue', edgecolor='black')
        plt.title('Entropy Distribution of Unselected Samples')
        plt.xlabel('Entropy')
        plt.ylabel('Number of Samples')
        plt.grid(axis='y', alpha=0.75)
        os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        plt_path = os.path.join(self.config.trainer.save_checkpoint_path, f"entropy_distribution_step_{self.global_continuous_step}.png")
        plt.savefig(plt_path)
        print(f"Entropy distribution histogram saved to {plt_path}")
        plt.close()

        return samples_with_entropy

    # === 新增：SGPO 训练子函数 ===
    def _sgpo_train_on_subset(self, subset_dataset: Dataset, round_num: int) -> str:
        """
        在给定的数据子集上运行完整的 SGPO 训练循环。
        """
        # 1. 基于子集创建新的 train_dataloader 
        # mini_rollout_batch_size是none
        if self.config.data.mini_rollout_batch_size is not None:
            train_batch_size = self.config.data.mini_rollout_batch_size
        else:
            train_batch_size = self.config.data.rollout_batch_size

        if self.config.data.shuffle:
            # 创建一个随机数生成器train_dataloader_generator。
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=subset_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=subset_dataset)

        # 构建一个batch size大小的新的train_dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=subset_dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )
        
        self.data_iterator = iter(self.train_dataloader)

        # 2. 计算此轮的训练步数
        if self.config.trainer.max_steps is not None:
             # 如果设置了 max_steps，则每轮都训练 max_steps max_steps =null
            self.training_steps = self.config.trainer.max_steps
        # elif 设置了 mini_rollout_batch_size，则根据 mini_batch_size 计算总步数
        # mini_rollout_batch_size =null
        elif self.config.data.mini_rollout_batch_size is not None:
            num_examples = len(self.train_dataloader) * self.config.data.mini_rollout_batch_size
            self.training_steps = num_examples // self.config.data.rollout_batch_size * self.config.trainer.total_epochs
        # 否则，按常规方式计算总步数 
        # len(self.train_dataloader) = 400/128  = 3 
        else:
            self.training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # 将 SGPO 优化器的步数设置为此轮的步数
        self.config.worker.actor.optim.training_steps = self.training_steps
        self.config.worker.critic.optim.training_steps = self.training_steps
        print(f"Round {round_num}: Starting SGPO training for {self.training_steps} steps.")

        # 3. 运行 SGPO 训练循环 (从原 fit 方法中平移过来)
        self.global_step = 0 # 重置 SGPO 训练步数
        main_tqdm = tqdm(range(self.training_steps), desc=f"SGPO Round {round_num}", position=0)

        while self.global_step < self.training_steps:
            self.global_step += 1
            
            #  累加全局连续步数 
            self.global_continuous_step += 1
            
            metrics, timing_raw = {}, {}
            
            with timer("step", timing_raw):
                # make a batch of data (从高熵子集中)
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # ... (计算 reward, old_log_probs, ref_log_probs, values, adv 的逻辑不变) ...
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    # [DEBUG]
                    if hasattr(old_log_probs, 'batch'):
                         print(f"[DEBUG] Old Log Probs keys: {list(old_log_probs.batch.keys())}")
                    batch = batch.union(old_log_probs)
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        # [DEBUG]
                        if hasattr(ref_log_probs, 'batch'):
                             print(f"[DEBUG] Ref Log Probs keys: {list(ref_log_probs.batch.keys())}")
                        batch = batch.union(ref_log_probs)
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        # [DEBUG]
                        # 我想知道这个batch.batch和reward_tensor, reward_metrics的对应，是每一个token都计算reward吗？
                        
                        print(f"[DEBUG] Reward: mean={reward_tensor.float().mean().item():.4f}, min={reward_tensor.min().item():.4f}, max={reward_tensor.max().item():.4f}")
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    # [DEBUG]
                    if 'advantages' in batch.batch:
                        adv = batch.batch['advantages']
                        print(f"[DEBUG] Advantages: mean={adv.float().mean().item():.4f}, min={adv.min().item():.4f}, max={adv.max().item():.4f}")
                    if 'returns' in batch.batch:
                        ret = batch.batch['returns']
                        print(f"[DEBUG] Returns: mean={ret.float().mean().item():.4f}, min={ret.min().item():.4f}, max={ret.max().item():.4f}")

                # ... (update critic, update actor 的逻辑不变) ...
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.non_tensor_batch))
                # 如果在 critic_warmup 期间，则跳过 actor 更新
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        # 这里ppo_epochs = 1
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.non_tensor_batch))

                # (SGPO 循环内部的验证和保存)
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    #  累加全局连续步数 
                    self.logger.log(data=metrics, step=self.global_continuous_step)
                
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint(round_num=round_num)

            # ... (日志记录逻辑不变) ...
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            # SGPO 循环内的日志
            #  使用累加全局连续步数 
            self.logger.log(data=metrics, step=self.global_continuous_step)
            main_tqdm.update()
            
        # SGPO 循环结束
        main_tqdm.close()
        
        # 在每轮 SGPO 训练结束时，保存最终模型
        print(f"SGPO training for round {round_num} finished. Saving final model...")
        final_actor_path = self._save_checkpoint(round_num=round_num)
        
        return final_actor_path # 返回最终模型的路径

    def fit(self):
        """
        执行主动学习循环。
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        
        # 1. 加载初始模型 (例如 SFT 模型)
        print("Loading initial checkpoint...")
        last_model_path = self._load_checkpoint()
        if last_model_path is None:
            print("[INFO] `_load_checkpoint` returned None. Proceeding with the model loaded during `init_workers` (SFT/Base model).")
        else:
            print(f"[INFO] `_load_checkpoint` successfully loaded SGPO checkpoint: {last_model_path}")
            
        # 2. 初始验证
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            print("Performing initial validation...")
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=0) # 记录在 step 0
            if self.config.trainer.val_only:
                return

        # 3. 准备主动学习
        # 假设 self.full_train_dataset 包含所有数据
        all_indices = list(range(len(self.full_train_dataset)))
        

        # 目标：加载 SFT 阶段的 string ID，并将它们映射到 full_train_dataset 的 integer 索引
        print("Building ID-to-Index map for Active Learning...")
        
        # 路径来自您的上下文
        # sft_id_path = "/path/to/checkpoint/saves/easy_r1/qwen2_5_vl_3b_urst_sgpo/SFT_models/selected_id.json"
        sft_id_path = os.path.join(os.path.dirname(self.config.trainer.load_checkpoint_path), "train_data_dict_round_init.json")
        
        # config.data.train_files 应该就是 all_data_path
        all_data_path = self.config.data.train_files 
        
        if not os.path.exists(all_data_path):
            raise FileNotFoundError(f"Cannot find the full dataset JSON at: {all_data_path}")

        # 1. 加载 SFT 阶段用过的 string ID 列表
        try:
            with open(sft_id_path, "r") as f:
                # 转换为 set 以便快速查询
                sft_ids = set(json.load(f))
            print(f"Loaded {len(sft_ids)} initial SFT IDs from {sft_id_path}")
        except Exception as e:
            print(f"WARNING: Could not load initial SFT IDs from {sft_id_path}. Starting with empty selection. Error: {e}")
            sft_ids = set()

        # 2. 加载完整的JSON数据集以构建 ID -> Index 映射
        try:
            with open(all_data_path, "r") as f:
                all_data_list = json.load(f)
        except Exception as e:
            raise IOError(f"Critical error: Could not load full dataset JSON from {all_data_path}. Error: {e}")

        # 3. 构建 ID -> Index 映射
        # (这个映射将 "1678..." -> 0, "1365..." -> 1, ...)
        id_to_index_map = {}
        for i, item in enumerate(all_data_list):
            if 'id' in item:
                id_to_index_map[item['id']] = i
        
        if len(id_to_index_map) != len(all_data_list):
            print("Warning: Some items in the full dataset may be missing an 'id'.")

        # 4. 将 SFT string ID 转换为 integer 索引
        selected_indices = set()
        for sft_id in sft_ids:
            if sft_id in id_to_index_map:
                selected_indices.add(id_to_index_map[sft_id])
            else:
                print(f"Warning: SFT ID {sft_id} not found in the full dataset.")
        
        print(f"Successfully mapped SFT IDs to {len(selected_indices)} initial integer indices.")
        
        al_config = self.config.active_learning

        # 4. 开始主动学习外循环
        for round_num in range(al_config.total_rounds):
            print(f"--- Starting Active Learning Round {round_num + 1} / {al_config.total_rounds} ---")
            
            # 4a. 识别未标记数据
            unselected_indices = [i for i in all_indices if i not in selected_indices]
            if not unselected_indices:
                print("No unselected data left. Stopping Active Learning.")
                break
                
            unselected_dataset = Subset(self.full_train_dataset, unselected_indices)
            print(f"Round {round_num}: Found {len(unselected_dataset)} unselected samples.")

            # 4b. 不确定性采样 - 修改为平衡采样
            # 返回 (original_index, majority_label, entropy) 列表
            samples_with_info = self._uncertainty_sampling(unselected_dataset)

            # 4c. 筛选 - 平衡选择 YES 和 NO 样本
            half_size = al_config.sample_size // 2
            remainder = al_config.sample_size % 2

            # 分离 YES 和 NO 样本
            yes_samples = [(idx, entropy) for idx, label, entropy in samples_with_info if label == "YES"]
            no_samples = [(idx, entropy) for idx, label, entropy in samples_with_info if label == "NO"]

            # 按熵降序排序(选择最不确定的)
            yes_samples.sort(key=lambda x: x[1], reverse=True)
            no_samples.sort(key=lambda x: x[1], reverse=True)

            # 选择样本
            query_samples = []
            query_samples.extend(yes_samples[:half_size + remainder])  # YES 样本(如果有余数,YES 多一个)
            query_samples.extend(no_samples[:half_size])  # NO 样本

            print(f"Selected {len(yes_samples[:half_size + remainder])} YES samples and {len(no_samples[:half_size])} NO samples")
            query_indices = [idx for idx, entropy in query_samples]
            # 打乱query_indices
            random.shuffle(query_indices)

            print(f"Round {round_num}: Selected {len(query_indices)} new samples for training.")
            selected_indices.update(query_indices) # 更新已选集合
            
            # 4d. 准备训练子集
            # URST 论文说只在 query_ids (新选的) 上训练
            sgpo_train_dataset = Subset(self.full_train_dataset, query_indices)
            
            # 4e. 在子集上进行 SGPO 训练
            trained_model_path = self._sgpo_train_on_subset(
                subset_dataset=sgpo_train_dataset, 
                round_num=round_num
            )
            
            # 4f. 加载新训练的模型，准备下一轮采样
            print(f"Round {round_num}: SGPO training complete. Loading new model from {trained_model_path}")
            self.actor_rollout_ref_wg.load_checkpoint(trained_model_path)
            # (同时加载 critic)
            if self.use_critic:
                critic_path = os.path.join(os.path.dirname(trained_model_path), "critic")
                if os.path.exists(critic_path):
                    self.critic_wg.load_checkpoint(critic_path)
                    
            # 4g. 验证新模型
            if self.val_reward_fn is not None:
                print(f"Round {round_num}: Validating new model...")
                val_metrics = self._validate()
                # 使用当前的全局累加步数
                self.logger.log(data=val_metrics, step=self.global_continuous_step)
                print(f"Round {round_num} Validation Metrics: {val_metrics}")
        
        print("Active Learning process finished.")
        print("--- Starting Final Evaluation on Test Sets ---")
        self._final_evaluation()
        print("--- Final Evaluation Complete ---")

    def _final_evaluation(self):
        """
        在训练结束后，对三个指定测试集运行最终评估。
        计算 Acc 和 F1，并保存结果到 JSON 文件。
        """
        print("[Final Evaluation] Starting final evaluation on test sets...")
        
        # 1. 获取配置
        try:
            # val_files 路径为 .../AITW_ID_traj_list.json
            # 我们需要它所在的目录 .../test_data/
            base_test_dir = self.config.data.val_files
            image_dir = self.config.data.image_dir
            answer_key = self.config.data.answer_key
        except Exception as e:
            print(f"[Final Evaluation] ERROR: Failed to get paths from config. {e}")
            return

        test_files = [
            "AITW_ID_traj_list.json",
            "AITW_OOD_traj_list.json",
            "AW_OOD_traj_list.json",
        ]
        
        all_results = {}
        all_acc = []
        all_f1 = []

        # 2. 循环遍历每个测试文件
        for file_name in test_files:
            file_path = os.path.join(base_test_dir, file_name)
            print(f"[Final Evaluation] Preparing to evaluate on file: {file_path}")
            if not os.path.exists(file_path):
                print(f"[Final Evaluation] WARNING: Test file not found, skipping: {file_path}")
                continue

            print(f"[Final Evaluation] Evaluating: {file_name}")

            # 3. 为每个测试文件创建 Dataset 和 DataLoader
            try:
                test_dataset = RLHFDataset(
                    data_path=file_path,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    prompt_key=self.config.data.prompt_key,
                    answer_key=answer_key,
                    image_key=self.config.data.image_key,
                    video_key=self.config.data.video_key,
                    image_dir=image_dir,
                    video_fps=self.config.data.video_fps,
                    max_prompt_length=self.config.data.max_prompt_length,
                    truncation="right",
                    format_prompt=self.config.data.format_prompt,
                    min_pixels=self.config.data.min_pixels,
                    max_pixels=self.config.data.max_pixels,
                    filter_overlong_prompts=False,
                )
                
                # 【关键修复1】：使用较小的 batch_size，并设置 drop_last=False
                eval_batch_size = min(8, len(test_dataset))  # 使用较小的 batch size
                
                test_dataloader = StatefulDataLoader(
                    dataset=test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=False,  # 【重要】不丢弃最后不完整的 batch
                )
                
                if len(test_dataloader) == 0:
                    print(f"[Final Evaluation] WARNING: DataLoader for {file_name} is empty. Skipping.")
                    continue

            except Exception as e:
                print(f"[Final Evaluation] ERROR: Failed to create DataLoader for {file_name}. {e}")
                import traceback
                traceback.print_exc()
                continue

            predictions = []
            ground_truths = []

            # 4. 运行推理
            self.actor_rollout_ref_wg.prepare_rollout_engine()
            for batch_idx, batch_dict in enumerate(tqdm(test_dataloader, desc=f"Testing {file_name}")):
                try:
                    # 【关键修复2】：在 pop 之前先保存 ground_truth
                    # 因为 ground_truth 在 from_single_dict 后存储在 non_tensor_batch 中
                    batch_ground_truths = batch_dict.get("ground_truth", None)
                    
                    if batch_ground_truths is None:
                        print(f"[Final Evaluation] WARNING: Batch {batch_idx} has no 'ground_truth' field. Skipping.")
                        continue
                    
                    # 转换为列表（可能是 numpy array）
                    if isinstance(batch_ground_truths, np.ndarray):
                        batch_ground_truths = batch_ground_truths.tolist()
                    elif not isinstance(batch_ground_truths, list):
                        batch_ground_truths = [batch_ground_truths]
                    
                    test_batch = DataProto.from_single_dict(batch_dict)
                    
                    gen_batch = test_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                    
                    # 使用贪婪解码进行确定性评估
                    gen_batch.meta_info = self.config.worker.rollout.val_override_config.copy()
                    gen_batch.meta_info["do_sample"] = False
                    gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
                    gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
                    gen_batch.meta_info["video_fps"] = self.config.data.video_fps
                    
                    # 【关键修复3】：处理 padding 以匹配 world_size
                    gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
                    test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

                    # 解码和解析
                    output_ids = test_output_gen_batch.batch["responses"]
                    raw_model_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    parsed_answers = [self.parse_answer_from_output(raw_output) for raw_output in raw_model_outputs]
                    
                    # 【关键修复4】：使用之前保存的 ground_truth
                    predictions.extend(parsed_answers)
                    ground_truths.extend(batch_ground_truths)
                    
                    # 调试信息
                    if batch_idx == 0:
                        print(f"[Final Evaluation] Sample prediction: {parsed_answers[0]}")
                        print(f"[Final Evaluation] Sample ground truth: {batch_ground_truths[0]}")
                
                except Exception as e:
                    print(f"[Final Evaluation] ERROR during batch {batch_idx} inference: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            self.actor_rollout_ref_wg.release_rollout_engine()

            # 5. 计算指标
            if not ground_truths:
                print(f"[Final Evaluation] ERROR: No ground truths were collected for {file_name}.")
                print(f"[Final Evaluation] Debug: predictions length = {len(predictions)}")
                continue

            print(f"[Final Evaluation] Collected {len(ground_truths)} samples for {file_name}")
            print(f"[Final Evaluation] Sample of ground truths: {ground_truths[:5]}")
            print(f"[Final Evaluation] Sample of predictions: {predictions[:5]}")
            
            # 【关键修复5】：确保标签格式一致（大写）
            ground_truths = [str(gt).upper() for gt in ground_truths]
            predictions = [str(pred).upper() for pred in predictions]
            
            # 过滤掉 UNKNOWN 的预测（可选）
            valid_indices = [i for i, (gt, pred) in enumerate(zip(ground_truths, predictions)) 
                            if gt in ["YES", "NO"] and pred in ["YES", "NO"]]
            
            # 记录 UNKNOWN 的输出
            unknown_indices = [i for i, (gt, pred) in enumerate(zip(ground_truths, predictions)) 
                            if gt in ["YES", "NO"] and pred == "UNKNOWN"]
            
            if unknown_indices:
                unknown_log_path = os.path.join(
                    self.config.trainer.save_checkpoint_path, 
                    f"unknown_predictions_{file_name.replace('.json', '')}.json"
                )
                unknown_samples = []
                for idx in unknown_indices:
                    unknown_samples.append({
                        "index": idx,
                        "ground_truth": ground_truths[idx],
                        "prediction": predictions[idx],
                        "raw_output": raw_model_outputs[idx] if idx < len(raw_model_outputs) else "N/A"
                    })
                
                try:
                    with open(unknown_log_path, "w") as f:
                        json.dump(unknown_samples, f, indent=4, ensure_ascii=False)
                    print(f"[Final Evaluation] Saved {len(unknown_indices)} UNKNOWN predictions to: {unknown_log_path}")
                except Exception as e:
                    print(f"[Final Evaluation] WARNING: Failed to save UNKNOWN predictions log. {e}")
            

                if not valid_indices:
                    print(f"[Final Evaluation] WARNING: No valid predictions for {file_name}.")
                    continue
            
            if not valid_indices:
                print(f"[Final Evaluation] WARNING: No valid predictions for {file_name}.")
                continue
                
            filtered_gts = [ground_truths[i] for i in valid_indices]
            filtered_preds = [predictions[i] for i in valid_indices]
            
            acc = accuracy_score(filtered_gts, filtered_preds)
            f1 = f1_score(filtered_gts, filtered_preds, average="macro", labels=["YES", "NO"])
            
            all_results[file_name] = {
                "accuracy": acc,
                "f1_score": f1,
                "total_samples": len(ground_truths),
                "valid_samples": len(filtered_gts),
                "unknown_predictions": len(predictions) - len(valid_indices)
            }
            all_acc.append(acc)
            all_f1.append(f1)
            
            print(f"[Final Evaluation] Results for {file_name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Valid samples: {len(filtered_gts)}/{len(ground_truths)}")

        # 6. 计算平均值并保存
        if all_acc:
            avg_acc = np.mean(all_acc)
            avg_f1 = np.mean(all_f1)
            all_results["average"] = {
                "accuracy_avg": avg_acc,
                "f1_score_avg": avg_f1
            }
            print(f"\n[Final Evaluation] Average Results:")
            print(f"  Average Accuracy: {avg_acc:.4f}")
            print(f"  Average F1 Score: {avg_f1:.4f}")
        else:
            print("[Final Evaluation] ERROR: No test files were successfully evaluated.")

        # 7. 保存到 JSON
        os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        save_path = os.path.join(self.config.trainer.save_checkpoint_path, "final_evaluation_results.json")
        try:
            with open(save_path, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"[Final Evaluation] Results saved to: {save_path}")
        except Exception as e:
            print(f"[Final Evaluation] ERROR: Failed to save results to JSON. {e}")
            import traceback
            traceback.print_exc()
    

        # self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        # self.global_step = 0
        # main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        # val_metrics: Optional[dict[str, Any]] = None

        # # load checkpoint before doing anything
        # self._load_checkpoint()
        # main_tqdm.update(self.global_step)

        # # perform validation before training
        # # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.val_before_train:
        #     val_metrics = self._validate()
        #     self.logger.log(data=val_metrics, step=self.global_step)
        #     if self.config.trainer.val_only:
        #         return
        
        # # TODO：构建训练数据集合，重新加载模型
        
        
        # self.data_iterator = iter(self.train_dataloader)
        
        # # 循环
        
        # ## 剩下的数据采样（vllm）
        # ## 底下的训练
        
        # while self.global_step < self.training_steps:
        #     self.global_step += 1
        
        #     metrics, timing_raw = {}, {}
        #     with timer("step", timing_raw):
        #         # make a batch of data
        #         with timer("gen", timing_raw):
        #             self.actor_rollout_ref_wg.prepare_rollout_engine()
        #             batch = self._make_batch_data(metrics=metrics)
        #             self.actor_rollout_ref_wg.release_rollout_engine()
        
        #         # balance the number of valid tokens on each dp rank.
        #         # NOTE: this breaks the order of data inside the batch.
        #         # Please take care when you implement group based adv computation such as GRPO and rloo
        #         self._balance_batch(batch, metrics=metrics)
        
        #         # compute global valid tokens
        #         batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        
        #         # compute reward
        #         if "token_level_scores" not in batch.batch:
        #             with timer("reward", timing_raw):
        #                 reward_ref = self.reward_fn.compute_reward.remote(batch)
        
        #         # recompute old_log_probs
        #         with timer("old", timing_raw):
        #             old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
        #             batch = batch.union(old_log_probs)
        
        #         # compute ref_log_probs
        #         if self.use_reference_policy:
        #             with timer("ref", timing_raw):
        #                 ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
        #                 batch = batch.union(ref_log_probs)
        
        #         # compute values
        #         if self.use_critic:
        #             with timer("values", timing_raw):
        #                 values = self.critic_wg.compute_values(batch)
        #                 batch = batch.union(values)
        
        #         with timer("adv", timing_raw):
        #             if "token_level_scores" not in batch.batch:
        #                 # get token level scores asynchronously
        #                 reward_tensor, reward_metrics = ray.get(reward_ref)
        #                 batch.batch["token_level_scores"] = reward_tensor
        #                 reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
        #                 metrics.update(reward_metrics)
        
        #             # apply kl penalty if available
        #             if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
        #                 # apply kl penalty to reward
        #                 batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
        #                 metrics.update(kl_metrics)
        #             else:
        #                 batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        
        #             # compute advantages, executed on the driver process
        #             batch = compute_advantage(
        #                 batch,
        #                 adv_estimator=self.config.algorithm.adv_estimator,
        #                 gamma=self.config.algorithm.gamma,
        #                 lam=self.config.algorithm.lam,
        #             )
        
        #         # update critic
        #         if self.use_critic:
        #             with timer("update_critic", timing_raw):
        #                 critic_output = self.critic_wg.update_critic(batch)
        
        #             critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
        #             metrics.update(critic_metrics)
        
        #         # update actor
        #         if self.config.trainer.critic_warmup <= self.global_step:
        #             with timer("update_actor", timing_raw):
        #                 actor_output = self.actor_rollout_ref_wg.update_actor(batch)
        
        #             actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
        #             metrics.update(actor_metrics)
        
        #         # validate
        #         if (
        #             self.val_reward_fn is not None
        #             and self.config.trainer.val_freq > 0
        #             and self.global_step % self.config.trainer.val_freq == 0
        #         ):
        #             with timer("validation", timing_raw):
        #                 val_metrics = self._validate()
        
        #             metrics.update(val_metrics)
        
        #         if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
        #             with timer("save_checkpoint", timing_raw):
        #                 self._save_checkpoint()

        #     # collect metrics
        #     num_gpus = self.resource_pool_manager.get_num_gpus()
        #     metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        #     metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        #     metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

        #     self.logger.log(data=metrics, step=self.global_step)
        #     main_tqdm.update()

        # # perform validation after training
        # if self.val_reward_fn is not None:
        #     if (
        #         val_metrics is None
        #         or self.config.trainer.val_freq <= 0
        #         or self.global_step % self.config.trainer.val_freq != 0
        #     ):
        #         val_metrics = self._validate()
        #         self.logger.log(data=val_metrics, step=self.global_step)

        #     print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        # if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
        #     self._save_checkpoint()
