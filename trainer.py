"""
SLTrust-DFL 核心训练器
支持所有聚合方法与 SAMA 对齐/时序 Byzantine 候选攻击。
"""
import copy
import os
import random
import torch
import numpy as np
import yaml
from pathlib import Path

from models.factory import build_model
from utils.data_loader import load_federated_dataset
from utils.topology import generate_topology
from utils.optim import build_optimizer, build_scheduler
from aggregators import (FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         BALANCEAggregator, SAMAAggregator, CONTRAAggregator,
                         CONTRAHardAggregator, SLTrustAggregator)
from attacks import ALIEAttack, LabelFlippingAttack, NoAttack, get_attack


# ──────────────────────────────────────────────────────────
# 聚合器工厂
# ──────────────────────────────────────────────────────────

def make_aggregator(method: str, cfg: dict):
    c = cfg.get(method, {})
    alpha = c.get('alpha', 0.5)
    if method == 'fedavg':
        return FedAvgAggregator(alpha=alpha)
    elif method == 'krum':
        return KrumAggregator(alpha=alpha, byzantine_ratio=c.get('byzantine_ratio', 0.2))
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator(alpha=alpha, trim_ratio=c.get('trim_ratio', 0.1))
    elif method == 'balance':
        return BALANCEAggregator(alpha=alpha, gamma=c.get('gamma', 0.5),
                                 kappa=c.get('kappa', 0.1))
    elif method == 'sama':
        return SAMAAggregator(alpha=alpha)
    elif method == 'contra':
        return CONTRAAggregator(alpha=alpha, beta=c.get('beta', 0.1))
    elif method == 'contra_hard':
        return CONTRAHardAggregator(alpha=alpha, beta=c.get('beta', 0.1),
                                    threshold=c.get('threshold', 0.1))
    elif method == 'sltrust':
        return SLTrustAggregator(alpha=alpha, beta=c.get('beta', 0.1),
                                 S=c.get('S', 5), L=c.get('L', 20), mode='full')
    elif method == 'sltrust_median':
        c = cfg.get('sltrust_median', cfg.get('sltrust', {}))
        return SLTrustAggregator(alpha=c.get('alpha', 0.5), beta=c.get('beta', 0.1),
                                 S=c.get('S', 3), L=c.get('L', 10), mode='median_bg')
    elif method == 'sltrust_no_ratio':
        c = cfg.get('sltrust_no_ratio', cfg.get('sltrust', {}))
        return SLTrustAggregator(alpha=c.get('alpha', 0.5), beta=c.get('beta', 0.1),
                                 S=c.get('S', 3), L=c.get('L', 10), mode='no_ratio')
    elif method == 'sltrust_short_only':
        c = cfg.get('sltrust_short_only', cfg.get('sltrust', {}))
        return SLTrustAggregator(alpha=c.get('alpha', 0.5), beta=c.get('beta', 0.1),
                                 S=c.get('S', 3), L=c.get('L', 10), mode='short_only')
    elif method == 'sltrust_long_only':
        c = cfg.get('sltrust_long_only', cfg.get('sltrust', {}))
        return SLTrustAggregator(alpha=c.get('alpha', 0.5), beta=c.get('beta', 0.1),
                                 S=c.get('S', 3), L=c.get('L', 10), mode='long_only')
    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────
# 攻击工厂
# ──────────────────────────────────────────────────────────

def make_attack(attack_name: str, cfg: dict, num_clients: int,
                num_byzantine: int, num_classes: int):
    kwargs = dict(cfg.get('attack', {}).get(attack_name, {}))
    if attack_name == 'alie':
        kwargs.update(num_byzantine=num_byzantine, num_clients=num_clients)
    elif attack_name == 'label_flipping':
        kwargs.setdefault('num_classes', num_classes)
    return get_attack(attack_name, **kwargs)


# ──────────────────────────────────────────────────────────
# 主训练器
# ──────────────────────────────────────────────────────────

class FederatedTrainer:

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config['experiment']['device']
                                   if torch.cuda.is_available() else 'cpu')

        num_clients = config['federated']['num_clients']
        byz_ratio = config['federated']['byzantine_ratio']
        num_byzantine = int(num_clients * byz_ratio)

        # 必须在数据划分和拓扑生成之前设置种子。这样每个独立运行进程中，
        # 相同 seed 的 method–attack 组合共享同一 partition/topology realization。
        seed = config['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 随机分配拜占庭节点（避免固定放在末尾导致拓扑偏差）
        rng = np.random.RandomState(seed)
        all_ids = list(range(num_clients))
        rng.shuffle(all_ids)
        self.byzantine_nodes = set(all_ids[:num_byzantine])
        self.honest_nodes = set(all_ids[num_byzantine:])
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine

        # 数据与拓扑均由配置驱动；同一 seed 固定 partition 和 graph realization。
        self.train_loaders, self.test_loader, dataset_spec = load_federated_dataset(
            config, self.device)
        self._in_channels = dataset_spec['in_channels']
        self._num_classes = dataset_spec['num_classes']

        topo = config['topology']['type']
        self.neighbors = generate_topology(config, num_clients, seed)

        print(f"Device: {self.device}  "
              f"Honest: {len(self.honest_nodes)}  Byzantine: {num_byzantine}  "
              f"Topology: {topo}")

    def train(self, method: str, attack_name: str) -> dict:
        """
        训练一个 (method, attack) 组合，返回 history dict。
        """
        print(f"\n--- {method.upper()} | attack={attack_name} ---")
        seed = self.cfg['experiment']['seed']
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 信誉型聚合器（CONTRA/SLTrust）每个节点需独立实例（各自维护对邻居的信誉状态）
        # 无状态聚合器（FedAvg/Krum等）所有节点共享一个实例即可
        _agg_proto = make_aggregator(method, self.cfg)
        needs_per_node_state = isinstance(_agg_proto, (CONTRAAggregator, CONTRAHardAggregator, SLTrustAggregator))
        if needs_per_node_state:
            aggregators = [make_aggregator(method, self.cfg) for _ in range(self.num_clients)]
            aggregator = aggregators[0]   # 用于 model_to_vector / load_from_vector 等静态方法
        else:
            aggregators = [_agg_proto] * self.num_clients
            aggregator = _agg_proto

        attack = make_attack(attack_name, self.cfg, self.num_clients,
                             self.num_byzantine, self._num_classes)

        # 所有客户端从同一初始模型出发；随后各自维护独立参数和优化器状态。
        initial_model = build_model(
            self.cfg, self._in_channels, self._num_classes).to(self.device)
        models = [copy.deepcopy(initial_model) for _ in range(self.num_clients)]
        del initial_model
        optimizers = [build_optimizer(m.parameters(), self.cfg) for m in models]
        eval_model = build_model(
            self.cfg, self._in_channels, self._num_classes).to(self.device)

        num_rounds = self.cfg['federated']['num_rounds']
        local_epochs = self.cfg['federated']['local_epochs']
        log_interval = self.cfg['logging']['log_interval']
        schedulers = [scheduler for scheduler in (
            build_scheduler(optimizer, self.cfg) for optimizer in optimizers)
            if scheduler is not None]

        history = {
            'test_acc': [], 'test_loss': [], 'consensus_error': [], 'tau_byz': [],
            # Per-round, per-observer edge evidence.  This is deliberately not
            # reconstructed from aggregate accuracy/tau logs: it is the only
            # formal source for phi, ratio, tau, gamma and honest-edge controls.
            'mechanism_trace': {'version': 1, 'rounds': []},
        }

        for t in range(num_rounds):
            # Byzantine 客户端永不聚合邻居。只有需要新鲜候选的攻击（标签翻转、
            # 时序 Gaussian 与方向扰动）才让 Byzantine 执行本地 SGD。
            local_models = []
            for i in range(self.num_clients):
                m = models[i]
                is_byzantine = i in self.byzantine_nodes
                is_training_node = (
                    not is_byzantine
                    or isinstance(attack, NoAttack)
                    or attack.trains_byzantine_locally
                )
                if is_training_node:
                    m.train()
                    for _ in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data = data.to(self.device, non_blocking=True)
                            target = target.to(self.device, non_blocking=True)
                            if is_byzantine and isinstance(attack, LabelFlippingAttack):
                                target = attack.flip_labels(target)
                            optimizers[i].zero_grad()
                            loss = torch.nn.functional.cross_entropy(m(data), target)
                            loss.backward()
                            optimizers[i].step()
                local_models.append(aggregator.model_to_vector(m).clone())

            # 攻击改写 Byzantine 广播候选。对动态攻击，这里处理的是刚由本轮
            # 本地 SGD 生成的 candidate；所有 Byzantine 都不参与邻居聚合。
            if isinstance(attack, ALIEAttack):
                honest_models = [local_models[i] for i in self.honest_nodes]
                malicious_model = attack.attack(honest_models)
                for byz_id in self.byzantine_nodes:
                    local_models[byz_id] = malicious_model.clone()
            elif not isinstance(attack, NoAttack):
                for byz_id in self.byzantine_nodes:
                    local_models[byz_id] = attack.attack(local_models[byz_id], t=t)

            # 拒绝非有限候选模型，避免单个 NaN/Inf 污染整个邻域。
            for sender_id, candidate_model in enumerate(local_models):
                if not torch.isfinite(candidate_model).all():
                    raise ValueError(
                        f"Non-finite candidate model broadcast by client {sender_id} at round {t + 1}")

            # 去中心化候选模型聚合。aggregate() 只处理邻居模型，final_update()
            # 在本地候选模型与邻居聚合模型之间执行一次 self-anchor 插值。
            all_models = torch.stack(local_models)
            updated_model_vecs = []
            round_edges = []
            for i in range(self.num_clients):
                own_model = local_models[i]
                nb_ids = self.neighbors[i]
                nb_models = [all_models[j] for j in nb_ids]

                participates = i in self.honest_nodes or isinstance(attack, NoAttack)
                if participates:
                    node_agg = aggregators[i]
                    # 信誉型聚合器需要 neighbor_ids 维护逐邻居历史。
                    if isinstance(node_agg, (CONTRAAggregator, CONTRAHardAggregator, SLTrustAggregator)):
                        agg_model, stats = node_agg.aggregate(
                            own_model, nb_models, neighbor_ids=nb_ids,
                            t=t, T=num_rounds, return_stats=True)
                    else:
                        agg_model, stats = node_agg.aggregate(
                            own_model, nb_models, t=t, T=num_rounds, return_stats=True)
                    # Reputation aggregators expose raw arrays aligned with
                    # neighbor_ids.  Preserve Byzantine and honest edges alike
                    # and compute gamma at each honest receiver from its actual
                    # normalized reputation mass.
                    if i in self.honest_nodes and all(
                            key in stats for key in ('neighbor_ids', 'phi', 'ratio', 'tau')):
                        total_tau = sum(float(value) for value in stats['tau'])
                        byz_tau = sum(
                            float(value) for sender, value in zip(stats['neighbor_ids'], stats['tau'])
                            if sender in self.byzantine_nodes
                        )
                        gamma = byz_tau / total_tau if total_tau > 0 else 0.0
                        for sender, phi, ratio, tau in zip(
                                stats['neighbor_ids'], stats['phi'], stats['ratio'], stats['tau']):
                            round_edges.append({
                                'receiver': int(i), 'sender': int(sender),
                                'is_byzantine': bool(sender in self.byzantine_nodes),
                                'phi': float(phi), 'ratio': float(ratio),
                                'tau': float(tau), 'gamma': float(gamma),
                            })
                    final_model = node_agg.final_update(own_model, agg_model)
                else:
                    final_model = own_model

                updated_model_vecs.append(final_model)

            # Keep an explicit empty edge list for non-reputation methods; a
            # formal mechanism plot will reject it instead of inventing values.
            history['mechanism_trace']['rounds'].append({'round': t + 1, 'edges': round_edges})

            for i, model_vec in enumerate(updated_model_vecs):
                aggregator.load_from_vector(models[i], model_vec)
            for scheduler in schedulers:
                scheduler.step()

            # ── 评估 ──────────────────────────────────────
            if (t + 1) % log_interval == 0:
                honest_list = [updated_model_vecs[i] for i in self.honest_nodes]
                honest_mat = torch.stack(honest_list)
                mean_vec = honest_mat.mean(dim=0)
                D_t = torch.mean(torch.norm(honest_mat - mean_vec, dim=1) ** 2).item()

                aggregator.load_from_vector(eval_model, mean_vec)
                eval_model.eval()
                correct = total = 0
                loss_sum = 0.0
                with torch.no_grad():
                    for data, target in self.test_loader:
                        data = data.to(self.device)
                        target = target.to(self.device)
                        out = eval_model(data)
                        loss_sum += torch.nn.functional.cross_entropy(
                            out, target, reduction='sum').item()
                        correct += out.argmax(1).eq(target).sum().item()
                        total += target.size(0)

                acc = 100.0 * correct / total
                loss_val = loss_sum / total
                history['test_acc'].append(acc)
                history['test_loss'].append(loss_val)
                history['consensus_error'].append(D_t)

                # 记录对拜占庭邻居的平均 τ（仅 CONTRA/SLTrust 有 _tau 属性）
                tau_byz_vals = []
                if needs_per_node_state:
                    for i in self.honest_nodes:
                        node_agg = aggregators[i]
                        if hasattr(node_agg, '_tau'):
                            for byz_id in self.byzantine_nodes:
                                if byz_id in node_agg._tau:
                                    tau_byz_vals.append(node_agg._tau[byz_id])
                history['tau_byz'].append(
                    float(np.mean(tau_byz_vals)) if tau_byz_vals else float('nan')
                )

                print(f"  Round {t+1:3d}/{num_rounds}  "
                      f"acc={acc:.2f}%  loss={loss_val:.4f}  D={D_t:.4f}")

        # 释放显存
        del models, eval_model, optimizers, aggregators
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
