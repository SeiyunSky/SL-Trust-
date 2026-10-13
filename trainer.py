"""
SLTrust-DFL 核心训练器
支持所有聚合方法 + 信誉专属攻击（含 Burn-in 的逐轮 t 传递）
"""
import copy
import os
import random
import torch
import numpy as np
import yaml
from pathlib import Path

from models import SimpleCNN
from utils.data_loader import load_mnist_gpu, load_mnist, load_cifar10_gpu, load_cifar10
from utils.topology import generate_ring_topology, generate_mesh_topology
from aggregators import (FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         BALANCEAggregator, SAMAAggregator, CONTRAAggregator,
                         CONTRAHardAggregator, SLTrustAggregator)
from attacks import (NoAttack, SignFlipAttack, GaussianAttack, ALIEAttack,
                     BurnInAttack, OscillatingAttack, SlowPoisonAttack,
                     BackdoorAttack, GradualFlipAttack, BurnInSlowAttack, get_attack)


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

def make_attack(attack_name: str, cfg: dict, num_clients: int, num_byzantine: int):
    attack_cfgs = cfg.get('attack', {})
    ac = attack_cfgs.get(attack_name, {})

    if attack_name == 'none':
        return NoAttack()
    elif attack_name == 'sign_flip':
        return SignFlipAttack(scale=ac.get('scale', 1.0))
    elif attack_name == 'gaussian':
        return GaussianAttack(std=ac.get('std', 5.0))
    elif attack_name == 'alie':
        return ALIEAttack(num_byzantine=num_byzantine, num_clients=num_clients)
    elif attack_name == 'burn_in':
        return BurnInAttack(warmup_rounds=ac.get('warmup_rounds', 40),
                            scale=ac.get('scale', 1.0))
    elif attack_name == 'oscillating':
        return OscillatingAttack(period_attack=ac.get('period_attack', 10),
                                 period_rest=ac.get('period_rest', 15),
                                 scale=ac.get('scale', 1.0))
    elif attack_name == 'slow_poison':
        return SlowPoisonAttack(scale=ac.get('scale', 0.1))
    elif attack_name == 'backdoor':
        return BackdoorAttack(lambda_bd=ac.get('lambda_bd', 0.3),
                              seed=ac.get('seed', 0))
    elif attack_name == 'gradual_flip':
        return GradualFlipAttack(warmup_rounds=ac.get('warmup_rounds', 40),
                                 ramp_rounds=ac.get('ramp_rounds', 20),
                                 scale=ac.get('scale', 1.0))
    elif attack_name == 'burn_in_slow':
        return BurnInSlowAttack(warmup_rounds=ac.get('warmup_rounds', 40),
                                scale=ac.get('scale', 0.3))
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


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

        # 数据
        data_dir = config['data']['data_dir']
        dataset = config['data'].get('dataset', 'mnist').lower()
        alpha = config['data']['non_iid_alpha']
        batch_size = config['federated']['batch_size']

        use_gpu_loader = torch.cuda.is_available()
        if dataset == 'cifar10':
            in_channels, num_classes = 3, 10
            if use_gpu_loader:
                self.train_loaders, self.test_loader = load_cifar10_gpu(
                    data_dir=data_dir, num_clients=num_clients,
                    alpha=alpha, batch_size=batch_size, device=str(self.device))
            else:
                self.train_loaders, self.test_loader = load_cifar10(
                    data_dir=data_dir, num_clients=num_clients,
                    alpha=alpha, batch_size=batch_size, num_workers=0)
        else:  # mnist
            in_channels, num_classes = 1, 10
            if use_gpu_loader:
                self.train_loaders, self.test_loader = load_mnist_gpu(
                    data_dir=data_dir, num_clients=num_clients,
                    alpha=alpha, batch_size=batch_size, device=str(self.device))
            else:
                self.train_loaders, self.test_loader = load_mnist(
                    data_dir=data_dir, num_clients=num_clients,
                    alpha=alpha, batch_size=batch_size, num_workers=0)
        self._in_channels = in_channels
        self._num_classes = num_classes

        # 拓扑
        topo = config['topology']['type']
        if topo == 'ring':
            self.neighbors = generate_ring_topology(num_clients)
        else:
            self.neighbors = generate_mesh_topology(
                num_clients, degree=config['topology'].get('degree', 4))

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

        attack = make_attack(attack_name, self.cfg,
                             self.num_clients, self.num_byzantine)

        # 所有客户端从同一初始模型出发；随后各自维护独立参数和优化器状态。
        initial_model = SimpleCNN(num_classes=self._num_classes,
                                  in_channels=self._in_channels).to(self.device)
        models = [copy.deepcopy(initial_model) for _ in range(self.num_clients)]
        del initial_model
        lr = self.cfg['optimizer']['lr']
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]
        eval_model = SimpleCNN(num_classes=self._num_classes,
                               in_channels=self._in_channels).to(self.device)

        num_rounds = self.cfg['federated']['num_rounds']
        local_epochs = self.cfg['federated']['local_epochs']
        log_interval = self.cfg['logging']['log_interval']

        history = {'test_acc': [], 'test_loss': [], 'consensus_error': [],
                   'tau_byz': []}   # 平均 τ (byzantine neighbors) per log step，仅信誉方法有值

        for t in range(num_rounds):
            # SAMA model-space backbone：每个节点先完成本地训练，随后广播候选模型
            # w_i^{t,+}。SLTrust 的相似度、信誉、幅度对齐和聚合均作用于候选模型。
            local_models = []
            for i in range(self.num_clients):
                m = models[i]
                m.train()
                for _ in range(local_epochs):
                    for data, target in self.train_loaders[i]:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        optimizers[i].zero_grad()
                        loss = torch.nn.functional.cross_entropy(m(data), target)
                        loss.backward()
                        optimizers[i].step()
                local_models.append(aggregator.model_to_vector(m).clone())

            # 恶意消息只替换 Byzantine 节点的广播候选模型，不覆盖其内部本地模型。
            broadcast_models = list(local_models)
            if isinstance(attack, ALIEAttack):
                honest_models = [local_models[i] for i in self.honest_nodes]
                malicious_model = attack.attack(honest_models)
                for byz_id in self.byzantine_nodes:
                    broadcast_models[byz_id] = malicious_model
            elif not isinstance(attack, NoAttack):
                for byz_id in self.byzantine_nodes:
                    broadcast_models[byz_id] = attack.attack(
                        local_models[byz_id], t=t)

            # 拒绝非有限广播，避免单个 NaN/Inf 污染整个邻域。
            for sender_id, broadcast_model in enumerate(broadcast_models):
                if not torch.isfinite(broadcast_model).all():
                    raise ValueError(
                        f"Non-finite candidate model broadcast by client {sender_id} at round {t + 1}")

            # 去中心化候选模型聚合。aggregate() 只处理邻居模型，final_update()
            # 在本地候选模型与邻居聚合模型之间执行一次 self-anchor 插值。
            all_models = torch.stack(broadcast_models)
            updated_model_vecs = []
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
                    final_model = node_agg.final_update(own_model, agg_model)
                else:
                    final_model = own_model

                updated_model_vecs.append(final_model)

            for i, model_vec in enumerate(updated_model_vecs):
                aggregator.load_from_vector(models[i], model_vec)

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
