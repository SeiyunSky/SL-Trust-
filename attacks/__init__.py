"""
攻击模块 — 聚焦信誉机制专属攻击

所有 attack() 输入和输出均为本地训练后的候选模型参数 w_i^{t,+}，
与 SAMA-DFL 的 model-space gossip 协议一致。

攻击列表：
  none          不篡改候选模型
  sign_flip     Sign Flipping（持续候选模型取反）
  gaussian      Gaussian Noise（持续候选模型扰动）
  alie          A Little Is Enough（基于诚实更新统计）
  burn_in       Burn-in / Reputation Farming（前K轮诚实后突然SF）
  oscillating   Oscillating（攻击/停止/攻击交替）
  slow_poison   Scaled Sign Flip（名称为历史兼容，并非隐蔽慢投毒）
  backdoor      Direction Perturbation Proxy（并非数据层标准后门）
"""
import torch
import numpy as np


class NoAttack:
    name = "NoAttack"

    def attack(self, own_vec, **kwargs):
        return own_vec


class SignFlipAttack:
    """将广播的候选模型整体取反并按 scale 缩放。"""
    name = "SignFlip"

    def __init__(self, scale=1.0):
        self.scale = scale

    def attack(self, own_vec, **kwargs):
        return -self.scale * own_vec


class GaussianAttack:
    """按候选模型 RMS 标定高斯噪声，避免噪声强度随维数失控。"""
    name = "Gaussian"

    def __init__(self, std=5.0, eps=1e-12):
        self.std = std
        self.eps = eps

    def attack(self, own_vec, **kwargs):
        rms = own_vec.norm() / max(own_vec.numel() ** 0.5, 1.0)
        if rms < self.eps:
            return own_vec
        return own_vec + torch.randn_like(own_vec) * (self.std * rms)


class ALIEAttack:
    """
    A Little Is Enough — Baruch et al., NeurIPS 2019
    恶意候选模型构造在诚实候选模型分布的统计置信区间边界
    """
    name = "ALIE"

    def __init__(self, num_byzantine: int, num_clients: int):
        self.num_byzantine = num_byzantine
        self.num_clients = num_clients

    def attack(self, honest_vecs, **kwargs):
        if isinstance(honest_vecs, list):
            mat = torch.stack(honest_vecs)
        else:
            mat = honest_vecs

        n, b = self.num_clients, self.num_byzantine
        s = int(np.floor(n / 2.0 + 1)) - b

        from scipy import stats as scipy_stats
        threshold = np.clip((n - b - s) / (n - b), 1e-6, 1 - 1e-6)
        z = float(scipy_stats.norm.ppf(threshold))

        mu = mat.mean(dim=0)
        std = mat.std(dim=0).clamp(min=1e-6)
        return mu - z * std


class BurnInAttack:
    """
    Burn-in / Reputation Farming 攻击（arXiv:2306.08669）
    前 warmup_rounds 轮行为诚实（积累信誉），之后切换为 Sign Flip
    """
    name = "BurnIn"

    def __init__(self, warmup_rounds: int = 40, scale: float = 1.0):
        self.warmup_rounds = warmup_rounds
        self.scale = scale

    def attack(self, own_vec, t: int = 0, honest_vecs=None, **kwargs):
        if t < self.warmup_rounds:
            return own_vec          # 诚实阶段：原样返回
        return -self.scale * own_vec    # 攻击阶段：候选模型取反


class OscillatingAttack:
    """
    振荡攻击：攻击 period_attack 轮，然后停止 period_rest 轮，交替循环
    测试双窗口是否被周期性重置诱导
    """
    name = "Oscillating"

    def __init__(self, period_attack: int = 10, period_rest: int = 15, scale: float = 1.0):
        self.period_attack = period_attack
        self.period_rest = period_rest
        self.scale = scale
        self._period = period_attack + period_rest

    def attack(self, own_vec, t: int = 0, **kwargs):
        phase = t % self._period
        if phase < self.period_attack:
            return -self.scale * own_vec    # 攻击阶段
        return own_vec                       # 休息阶段（看起来诚实）


class SlowPoisonAttack:
    """
    历史兼容名称。实际执行候选模型的 scaled sign flip。
    在采用幅度对齐的聚合器中，正 scale 的绝对大小会被消除，因此它与
    满幅 sign flip 方向等价，不能解释为独立的隐蔽慢投毒攻击。
    """
    name = "SlowPoison"

    def __init__(self, scale: float = 0.1):
        self.scale = scale

    def attack(self, own_vec, **kwargs):
        return -self.scale * own_vec


class GradualFlipAttack:
    """
    Gradual Flip 攻击——warmup 期诚实，之后在 ramp_rounds 轮内将翻转幅度从 0 线性
    过渡到 scale，之后持续满幅翻转。
    比 BurnIn 更难防：ratio 变化更缓慢，测 SLTrust 对慢速翻转的敏感度。
    """
    name = "GradualFlip"

    def __init__(self, warmup_rounds: int = 40, ramp_rounds: int = 20, scale: float = 1.0):
        self.warmup_rounds = warmup_rounds
        self.ramp_rounds = ramp_rounds
        self.scale = scale

    def attack(self, own_vec, t: int = 0, **kwargs):
        if t < self.warmup_rounds:
            return own_vec
        ramp_t = t - self.warmup_rounds
        if ramp_t < self.ramp_rounds:
            alpha = ramp_t / self.ramp_rounds   # 0→1 线性增大
            return (1.0 - 2.0 * alpha * self.scale) * own_vec
        return -self.scale * own_vec


class BurnInSlowAttack:
    """
    历史兼容名称。warmup 后对候选模型执行 scaled sign flip。
    在幅度对齐聚合器中 scale 大小会被消除，不能解释为独立慢投毒。
    """
    name = "BurnInSlow"

    def __init__(self, warmup_rounds: int = 40, scale: float = 0.3):
        self.warmup_rounds = warmup_rounds
        self.scale = scale

    def attack(self, own_vec, t: int = 0, **kwargs):
        if t < self.warmup_rounds:
            return own_vec
        return -self.scale * own_vec



class BackdoorAttack:
    """
    方向扰动代理：对本地候选模型叠加固定随机方向。

    该实现没有数据触发器、目标标签或 ASR，因此不是标准后门攻击；保留
    BackdoorAttack 类名仅用于兼容已有配置。
    """
    name = "Backdoor"

    def __init__(self, lambda_bd: float = 0.3, seed: int = 0):
        self.lambda_bd = lambda_bd
        self._trigger = None
        self._seed = seed

    def _get_trigger(self, vec):
        if self._trigger is None or self._trigger.shape != vec.shape:
            rng = torch.Generator(device=vec.device)
            rng.manual_seed(self._seed)
            t = torch.randn(vec.shape, generator=rng, device=vec.device)
            self._trigger = t / (t.norm() + 1e-8)
        return self._trigger.to(vec.device)

    def attack(self, own_vec, **kwargs):
        trigger = self._get_trigger(own_vec)
        # 保留大部分真实更新方向，混入固定随机方向
        mal_vec = (1.0 - self.lambda_bd) * own_vec + self.lambda_bd * trigger * own_vec.norm()
        return mal_vec


ATTACK_REGISTRY = {
    'none': NoAttack,
    'sign_flip': SignFlipAttack,
    'gaussian': GaussianAttack,
    'alie': ALIEAttack,
    'burn_in': BurnInAttack,
    'oscillating': OscillatingAttack,
    'slow_poison': SlowPoisonAttack,
    'backdoor': BackdoorAttack,
    'gradual_flip': GradualFlipAttack,
    'burn_in_slow': BurnInSlowAttack,
}


def get_attack(name: str, **kwargs):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**kwargs)
