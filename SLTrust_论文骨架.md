# SLTrust-DFL 论文骨架（投稿版）

> **文档定位**：这是论文的论证骨架，不是结果预告。实验结果尚未汇总的地方统一使用 `[待填]`，不得预写“显著优于”“全面领先”等结论。
>
> **核心主线**：去中心化信誉聚合使用历史平滑来抵抗短期噪声，但历史记忆也会造成信誉撤销迟滞。攻击者可以先表现正常、积累信誉，再切换为恶意行为。SLTrust 比较邻居的近期行为与**不重叠的历史背景**，仅在近期表现下降时加速遗忘旧信誉，从而缩短攻击切换后的高权重暴露期。
>
> **论文边界**：本文研究“能导致方向相似度可观察下降的行为切换”。不声称防御所有 Byzantine 攻击，也不声称能够检测保持高余弦相似度的后门或低角度投毒。

---

## 0. 投稿前统一约定

### 0.1 术语

全文统一使用：

- `round-start model`：节点本轮本地训练前的参数向量 \(w_i^t\)；
- `local candidate model`：节点完成本轮本地训练后的参数向量 \(v_i^t=w_i^{t,+}\)；
- `neighbor candidate model`：邻居广播的候选模型 \(\widetilde v_j^t\)；
- `instantaneous agreement score`：当轮方向一致性分数 \(\varphi_{ij}^t\)；
- `reputation`：跨轮维护的信誉 \(\tau_{ij}^t\)；
- `recent window`：最近 \(S\) 个观测；
- `historical background window`：历史队列中排除最近 \(S\) 个观测后的部分。

当前代码采用与 SAMA-DFL 一致的 **model-space gossip**：攻击、相似度、幅度对齐和聚合均作用于本地训练后的候选模型 \(v_i^t\)，不是某个 batch 的原始梯度。旧的累计更新 \(\Delta w\) 实现及其结果不得混入正式实验。

### 0.2 方法名称

在正式核验来源之前，论文中把代码里的 `CONTRA` 写为：

> **EMA reputation baseline**（代码标识为 `contra`）

不得把 Fung et al. 的 *Mitigating Sybils in Federated Learning Poisoning* 当作 CONTRA 来源；该论文对应 FoolsGold。只有找到真正与当前 EMA 公式及 DFL 设置一致的原始来源后，才能恢复 `CONTRA` 这一论文名称。

### 0.3 三层结论强度

- **理论可证明**：ratio 单向性、条件性加速遗忘、稳定行为时退化为 EMA；
- **实验待验证**：在 Burn-in / Oscillating 下是否提高攻击期精度、缩短信誉撤销时间；
- **当前不主张**：继承 SAMA 的全局收敛界、普适 Byzantine 鲁棒性、后门防御能力、“首个”工作。

---

# 论文标题候选

## 推荐标题

**SLTrust: Accelerating Reputation Revocation against Behavior-Switching Attacks in Decentralized Federated Learning**

这个标题准确描述机制，不使用未经证实的“robust against all Byzantine attacks”。

## 备选标题

1. **History Can Hurt: Mitigating Reputation Inertia in Decentralized Federated Learning**
2. **SLTrust: Short–Long Behavioral Contrast for Reputation-Aware Decentralized Federated Learning**
3. **Revoking Earned Trust: Defending Decentralized Federated Learning against Reputation Farming**

若实验最终只在 Burn-in 上优势明显，优先使用标题 3；若多个 behavior-switching 场景均稳定，使用推荐标题。

---

# Abstract（结果回来后填写数值）

按以下五句结构写，不超过 180–220 词：

1. **背景**：Reputation-weighted decentralized federated learning smooths historical observations to reduce the impact of transient noise.
2. **问题**：However, this memory creates reputation inertia: a Byzantine neighbor can behave benignly to accumulate trust and then retain non-negligible aggregation weight after switching to malicious updates.
3. **方法**：We propose SLTrust, which contrasts a short recent window with a disjoint historical background and multiplicatively discounts only the inherited reputation when recent agreement deteriorates.
4. **分析**：We show that SLTrust never amplifies inherited reputation relative to an EMA baseline and achieves a smaller memory factor whenever the recent-to-background ratio is bounded below one.
5. **实验**：Across `[数据集/拓扑/seed 数]`, SLTrust reduces `[time-to-neutralize]` by `[待填]` and improves `[attack-window accuracy]` by `[待填]` over the EMA reputation baseline under behavior-switching attacks, while incurring `[待填]` clean-performance change. Results on gradual or direction-preserving attacks identify the method's detection boundary.

禁止在结果未支持时写：`consistently outperforms all baselines`、`universally robust`、`eliminates reputation farming`。

---

# 1. Introduction

## 1.1 背景：DFL 中的本地信任决策

- 去中心化联邦学习没有中心服务器，每个节点只与图上的邻居交换模型参数；
- 每个诚实节点必须基于有限的局部观测决定如何融合邻居模型；
- 单轮相似度容易受 Non-IID 和随机训练噪声影响，因此一种自然做法是维护跨轮信誉并用 EMA 平滑；
- 历史平滑改善稳定性的同时，也降低了对行为切换的响应速度。

**本段落目标**：从“历史记忆的稳定性—响应性矛盾”引出问题，而不是泛称所有 DFL 防御都只看单轮信息。

## 1.2 关键漏洞：Reputation Inertia

定义一个行为切换攻击：

1. 攻击者在 warm-up 阶段发送正常模型向量；
2. 其信誉逐渐升高；
3. 在切换轮 \(K+1\) 后发送方向异常的模型；
4. 标准 EMA 仍保留此前积累的信誉。

若 EMA 更新为：

\[
\tau_{ij}^t=\beta\varphi_{ij}^t+(1-\beta)\tau_{ij}^{t-1},
\]

且切换前 \(\tau_{ij}^{K}\approx1\)、切换后 \(\varphi_{ij}^{K+1}=0\)，则：

\[
\tau_{ij}^{K+1}\approx1-\beta.
\]

这说明问题不是 EMA 永远不能降低恶意节点信誉，而是**撤销速度可能慢于攻击造成损害的速度**。

将该漏洞称为 `reputation inertia`。`reputation farming` 是利用此惯性的具体攻击策略：先积累、后消费信誉。

## 1.3 核心洞察

一个节点从“长期正常”突然变成“近期异常”时：

- 近期均值下降；
- 不包含近期样本的历史背景仍然较高；
- 两者比值可以作为行为下降强度的信号。

如果近期表现提升，比值可能大于 1，但不应因此加速恢复信誉。因此 SLTrust 使用：

\[
r_{ij}^t=\min\left(1,
\frac{\bar\varphi_{S,ij}^t}
{\bar\varphi_{\mathrm{bg},ij}^t+\epsilon}\right).
\]

该因子只乘在旧信誉项上：

\[
\tau_{ij}^t=\beta\varphi_{ij}^t+
(1-\beta)\tau_{ij}^{t-1}r_{ij}^t.
\]

因此 SLTrust 的本质是 **conditional reputation revocation**，不是通用攻击检测器。

## 1.4 研究问题

论文围绕三个明确问题组织：

- **RQ1 — Vulnerability**：EMA reputation 在行为切换后是否存在可测量的信誉撤销迟滞？
- **RQ2 — Mechanism**：非重叠 recent/background 对比是否能缩短恶意信誉撤销时间，且单向截断是否避免异常加速恢复？
- **RQ3 — Utility and boundary**：该改进能否提升攻击期性能而不明显损害无攻击训练？在哪些渐进或方向保持型攻击下失效？

## 1.5 贡献

在实验结果确认后，贡献写成以下形式：

1. **问题贡献**：形式化 DFL 信誉聚合中的 reputation inertia，并构造行为切换威胁模型，区分“持续恶意”与“先积累后攻击”。
2. **方法贡献**：提出 SLTrust，以近期窗口和非重叠历史背景的比值对旧信誉实施单向折扣；该方法仅增加每条邻接边 \(O(L)\) 的标量状态。
3. **分析贡献**：证明 SLTrust 不会相对 EMA 放大旧信誉，并在近期—背景比值持续低于 1 时具有更小的信誉记忆因子；同时给出该保证不适用于方向相似度不下降攻击的边界。
4. **实证贡献**：在 `[待填：数据集、拓扑、seed]` 上，从模型效用和信誉动力学两个层面评估行为切换攻击，并通过 `[待填：消融]` 分离非重叠背景和单向截断的作用。

不要使用“首个”作为贡献成立的前提。

---

# 2. Related Work

> 本节最终必须逐篇核验原论文。本骨架只规定比较维度，不替代文献调查。

## 2.1 Byzantine-Robust Federated and Distributed Learning

介绍 Krum、coordinate-wise trimmed mean、median 等中心化或分布式鲁棒聚合。明确：

- 它们主要基于同一轮收到的一组更新进行鲁棒统计；
- 原始理论通常要求足够大的参与者集合及 Byzantine 数量条件；
- 本文 degree=4 的局部 DFL 使用属于 localized adaptation，不能声称满足原始 Krum 理论条件。

## 2.2 Byzantine-Robust Decentralized Learning

只纳入经核实确实属于 decentralized Byzantine learning 的工作，按以下维度比较：

- 是否需要中心服务器或可信根数据；
- 每节点观察局部邻居还是全体客户端；
- 单轮还是跨轮状态；
- 硬筛选、裁剪还是软加权；
- 是否显式处理行为切换。

SAMA 是本课题前期方法时，应明确其公开状态和可获取性；未正式发表的材料不能作为外部读者必须接受的理论基础。

## 2.3 Historical and Reputation-Based Defenses

讨论：

- EMA 或累计信誉的稳定作用；
- 历史梯度/更新相似性方法；
- FoolsGold 的跨客户端 Sybil 相似性目标；
- 与 SLTrust 的区别：SLTrust 比较**同一邻接边随时间的近期行为与历史背景**，不是检测多个客户端之间的协同行为。

代码中的 `CONTRA` 在来源核验前统一称 EMA reputation baseline。

## 2.4 Adaptive, On–Off, and Reputation-Farming Attacks

区分：

- 持续攻击；
- abrupt behavior switching / burn-in；
- on–off / oscillating；
- gradual drift；
- direction-preserving poisoning / backdoor。

本节结尾明确研究空缺：历史信誉机制如何在局部 DFL 中撤销已积累信誉，以及该撤销机制的可观测性边界。

---

# 3. Problem Formulation

## 3.1 Decentralized Learning Model

系统由无向图 \(G=(V,E)\) 表示，\(|V|=n\)。所有节点从共同初始化出发。节点 \(i\) 在轮次 \(t\) 从 \(w_i^t\) 开始进行多步本地训练，得到候选模型：

\[
v_i^t:=w_i^{t,+}=w_i^t-\eta_t g_i^t.
\]

节点从邻居集合 \(\mathcal N_i\) 接收广播候选模型 \(\widetilde v_j^t\)。诚实节点先计算邻居聚合模型 \(A_i^t\)，再执行：

\[
w_i^{t+1}=\alpha v_i^t+(1-\alpha)A_i^t.
\]

Byzantine 节点可篡改其广播候选模型，但恶意广播与其内部本地模型状态分离；诚实节点不知道哪些邻居为 Byzantine。

必须在正文说明：实验评估模型使用诚实节点参数均值，仅用于离线测量，不参与训练或通信。

## 3.2 Instantaneous Agreement

节点 \(i\) 对邻居 \(j\) 的当轮一致性定义为：

\[
\varphi_{ij}^t=
\max\left(0,
\frac{\langle v_i^t,\widetilde v_j^t\rangle}
{\|v_i^t\|_2\|\widetilde v_j^t\|_2+\epsilon}
\right).
\]

说明 ReLU 的含义：负方向相似度直接视为零。零范数向量也记为零信誉观测。

## 3.3 Threat Model

### 攻击者能力

- 控制固定 Byzantine 节点集合 \(V_B\)；
- Byzantine 节点可任意修改其广播的候选模型；
- 在 Burn-in 中，攻击者可选择切换轮 \(K\)；
- ALIE 实现使用全部诚实向量统计量，因此属于较强的 omniscient 攻击设置，必须单独说明；
- 默认不考虑身份伪造、通信丢包、异步消息和拓扑篡改，除非实验显式实现。

### 核心攻击：Burn-in / Reputation Farming

\[
\widetilde v_j^t=
\begin{cases}
v_j^t, & t<K,\\
\mathcal A(v_j^t), & t\ge K.
\end{cases}
\]

核心实验中的 \(\mathcal A\) 是 sign flip。攻击者在 \(t<K\) 表现正常，因此在任何仅基于历史行为的系统中都会积累信誉；论文研究的是切换后的撤销迟滞，而不是事前识别尚未作恶的节点。

### 边界攻击

Gradual drift 和 direction-preserving perturbation 用于测量方法边界。若攻击不导致 \(\varphi\) 相对历史背景下降，SLTrust 没有可供检测的信号。

## 3.4 Reputation Inertia Metrics

定义后续实验直接使用的指标：

- **Time-to-neutralize**：攻击开始后，Byzantine 邻接边平均信誉首次低于阈值 \(\theta\) 的轮数；
- **Residual trust area**：攻击窗口内 Byzantine reputation 曲线面积；
- **Trust separation**：诚实边与 Byzantine 边平均信誉之差；
- **Attack-window utility**：攻击开始后固定窗口内的平均测试精度。

阈值 \(\theta\) 只作为摘要，应同时提供完整信誉曲线或多个阈值的敏感性。

---

# 4. SLTrust

## 4.1 Design Goals

- **G1 Fast revocation**：近期行为显著差于历史时，加速移除旧信誉；
- **G2 No accelerated recovery**：近期行为改善时，不因比值大于 1 而比 EMA 更快恢复；
- **G3 Conservative cold start**：没有独立背景时，不声称检测到行为变化；
- **G4 Locality**：只使用节点已有的邻居更新和每条边的本地历史，不增加通信量。

## 4.2 Disjoint Recent–Background Contrast

对每条有向观察边 \((i,j)\)，节点 \(i\) 保存最近最多 \(L\) 个 \(\varphi_{ij}\)。要求：

\[
1\le S<L.
\]

定义：

\[
\mathcal R_t=\{\max(1,t-S+1),\ldots,t\},
\]

\[
\mathcal B_t=\{\max(1,t-L+1),\ldots,t-S\}.
\]

\(\mathcal R_t\cap\mathcal B_t=\varnothing\)。当背景非空时：

\[
\bar\varphi_{S,ij}^t=
\frac{1}{|\mathcal R_t|}\sum_{r\in\mathcal R_t}\varphi_{ij}^r,
\]

\[
\bar\varphi_{\mathrm{bg},ij}^t=
\frac{1}{|\mathcal B_t|}\sum_{r\in\mathcal B_t}\varphi_{ij}^r.
\]

非重叠非常关键：若长窗口包含 recent，当前下降会同时降低分子和分母，削弱对行为切换的对比。

## 4.3 One-Sided Reputation Discount

\[
r_{ij}^t=
\begin{cases}
1, & \mathcal B_t=\varnothing,\\
\min\left(1,
\frac{\bar\varphi_{S,ij}^t}
{\bar\varphi_{\mathrm{bg},ij}^t+\epsilon}
\right), & \mathcal B_t\neq\varnothing.
\end{cases}
\]

信誉初始化和更新为：

\[
\tau_{ij}^1=\varphi_{ij}^1,
\]

\[
\tau_{ij}^t=\beta\varphi_{ij}^t+
(1-\beta)\tau_{ij}^{t-1}r_{ij}^t,
\qquad t>1.
\]

解释：

- recent 低于 background：\(r<1\)，额外折扣旧信誉；
- recent 不低于 background：\(r=1\)，退化为 EMA；
- 背景为空：\(r=1\)，冷启动时不实施变化检测。

## 4.4 Magnitude Alignment and Aggregation

邻居候选模型先对齐到自身候选模型范数：

\[
\widehat v_{ij}^{t}=
\|v_i^t\|_2
\frac{\widetilde v_j^t}{\|\widetilde v_j^t\|_2}.
\]

随后：

\[
A_i^t=
\frac{\sum_{j\in\mathcal N_i}\tau_{ij}^t\widehat v_{ij}^{t}}
{\sum_{j\in\mathcal N_i}\tau_{ij}^t}.
\]

若没有正信誉且非零范数的邻居，令 \(A_i^t=v_i^t\)，即本轮不接纳邻居信息。最终：

\[
w_i^{t+1}=\alpha v_i^t+(1-\alpha)A_i^t.
\]

说明：幅度对齐与 self-anchor 并非 SLTrust 的时间机制创新，应标注其来源或作为公共协议组件，不把它们重复计为本文贡献。

## 4.5 Algorithm 1

```text
Input: own candidate model v_i, broadcast neighbor models {v_j},
       edge histories H_ij, reputations tau_ij,
       1 <= S < L, beta, epsilon

for each neighbor j do
    phi_ij <- max(0, cosine(v_i, v_j))
    append phi_ij to H_ij and retain at most L entries

    recent <- last min(S, |H_ij|) entries of H_ij
    background <- all entries of H_ij except recent

    if background is empty then
        ratio <- 1
    else
        ratio <- min(1, mean(recent) / (mean(background) + epsilon))
    end if

    if tau_ij is uninitialized then
        tau_ij <- phi_ij
    else
        tau_ij <- beta * phi_ij + (1-beta) * tau_ij * ratio
    end if

    align v_j to ||v_i||
end for

A_i <- trust-weighted mean of valid aligned neighbor candidates
return alpha * v_i + (1-alpha) * A_i
```

## 4.6 Complexity

设模型维度为 \(p\)、节点度数为 \(d_i\)：

- 余弦和幅度对齐：\(O(d_i p)\)，与已有相似度聚合一致；
- 朴素窗口统计：每轮 \(O(d_iL)\)；
- 状态存储：\(O(d_iL)\) 个标量；
- 额外通信：0。

如果实现使用滚动和，可将窗口统计降为 \(O(d_i)\)，但当前代码使用 deque 后直接求和，论文应报告实际实现，而不是未实现的复杂度。

---

# 5. Analysis of Reputation Revocation

> 本节分析信誉动力学，不声称自动获得全局 DFL 优化收敛。若后续希望加入收敛定理，必须独立证明时变归一化权重下的聚合误差与诚实权重分母下界。

## 5.1 Basic Range

若 \(0\le\varphi_{ij}^t\le1\)、\(0<\beta\le1\)、\(0\le r_{ij}^t\le1\)，且 \(\tau_{ij}^1=\varphi_{ij}^1\)，则归纳可得：

\[
0\le\tau_{ij}^t\le1.
\]

证明：两个非负项之和非负，且：

\[
\tau_{ij}^t
\le\beta+(1-\beta)=1.
\]

## 5.2 Proposition 1: No Amplification of Inherited Reputation

设 SLTrust 和 EMA 在轮次 \(t-1\) 有相同信誉 \(\tau^{t-1}\)，并观察到相同 \(\varphi^t\)。因为 \(r^t\le1\)：

\[
\tau_{\mathrm{SL}}^t
=
\beta\varphi^t+(1-\beta)\tau^{t-1}r^t
\le
\beta\varphi^t+(1-\beta)\tau^{t-1}
=
\tau_{\mathrm{EMA}}^t.
\]

注意该命题比较的是**同一轮、相同前态**下的更新算子；若两种方法此前信誉已经不同，需要使用递推比较或下面的上界，不能跳步声称任意时刻都严格更小。

## 5.3 Proposition 2: Conditional Faster Forgetting

假设从轮次 \(t+1\) 起连续 \(m\) 轮满足：

\[
\varphi^{t+s}\le\bar\varphi,
\qquad
r^{t+s}\le\rho<1,
\quad s=1,\ldots,m.
\]

令 \(q=(1-\beta)\rho\)，则：

\[
\tau^{t+m}
\le
q^m\tau^t+
\frac{\beta\bar\varphi(1-q^m)}{1-q}.
\]

证明由一阶递推展开得到。旧信誉的记忆因子由 EMA 的 \((1-\beta)\) 变为 \((1-\beta)\rho\)。

**解释边界**：这是条件保证，不是对所有攻击的保证。若 \(r\approx1\) 或 \(\varphi\) 始终很高，SLTrust 不会获得额外遗忘。

## 5.4 Proposition 3: Conservative Behavior under Non-degradation

如果：

\[
\bar\varphi_S^t\ge\bar\varphi_{\mathrm{bg}}^t,
\]

则 \(r^t=1\)，SLTrust 当轮退化为标准 EMA。因此比值机制不会在近期表现不低于历史背景时额外惩罚旧信誉。

这不等于“诚实节点永不被惩罚”。Non-IID、局部漂移或训练震荡也可能使诚实邻居近期相似度下降；该误惩罚风险必须由 no-attack 与不同 heterogeneity 实验测量。

## 5.5 Burn-in Example

假设攻击前窗口内 \(\varphi=1\)，攻击后为 0，且 \(S=3\)。第一个攻击观测进入后：

\[
\bar\varphi_S=\frac{1+1+0}{3}=\frac23,
\qquad
\bar\varphi_{\mathrm{bg}}\approx1,
\]

所以 \(r\approx2/3\)，而不是 0。随着更多攻击观测填满 recent，ratio 才继续下降。这一例子说明：

- SLTrust 应被描述为“加速撤销”，不是“一轮归零”；
- 速度依赖 \(S\)、\(\beta\) 和攻击后的相似度；
- time-to-neutralize 是比最终精度更直接的机制指标。

## 5.6 Why a Global Convergence Claim Is Deferred

SLTrust 的实际聚合权重为：

\[
a_{ij}^t=\frac{\tau_{ij}^t}{\sum_{k\in\mathcal N_i}\tau_{ik}^t}.
\]

要证明全局收敛，至少还需控制：

- 诚实权重和的正下界；
- Byzantine 加权误差；
- 各节点历史相关权重造成的时变混合矩阵；
- Non-IID 下诚实相似度下降导致的误惩罚；
- 网络连通性和共识误差。

仅由 \(\tau\in[0,1]\) 不能推出与 SAMA 相同的 \(\kappa\) 或收敛速率。因此当前论文不把“继承 SAMA 收敛定理”列为贡献。未来若补出完整证明，可作为增强，而不是当前论文成立的前提。

---

# 6. Experimental Methodology

## 6.1 Claims–Evidence Matrix

| Claim | 主要对比 | 主要攻击 | 主要指标 | 所需图表 |
|---|---|---|---|---|
| EMA 存在撤销迟滞 | EMA baseline vs. No Attack trajectory | Burn-in | residual trust area、time-to-neutralize | trust 曲线 |
| SLTrust 加速撤销 | SLTrust vs. EMA baseline | Burn-in | time-to-neutralize、attack-window accuracy | 曲线 + 统计表 |
| 单向截断避免加速恢复 | Full vs. No-clip | Oscillating | rest-phase reputation recovery | 消融图 |
| 非重叠背景提供更强对比 | Full vs. Overlap-bg | Burn-in | ratio、trust decay | 消融图 |
| 正常训练代价可控 | SLTrust vs. EMA/FedAvg | No Attack | final / mean accuracy、consensus error | clean 表 |
| 方法存在可观测性边界 | SLTrust vs. EMA | Gradual / direction-preserving | \(\varphi\)、ratio、utility | limitation 图 |

如果当前正在运行的实验没有 `No-clip` 或 `Overlap-bg`，先如实报告现有 `no_ratio/short_only/long_only`；论文定稿前再补上述两项最有解释力的消融，不需要中断当前任务。

## 6.2 Experimental Setup

以实际配置文件为准，当前主配置应写为：

### MNIST

- 20 clients；
- Byzantine ratio = 20%；
- mesh topology，degree = 4；
- Dirichlet \(\alpha_{data}=0.5\)；
- 100 communication rounds；
- local epochs = 1，batch size = 128，learning rate = 0.01。

### CIFAR-10

- 20 clients；
- Byzantine ratio = 20%；
- mesh topology，degree = 4；
- Dirichlet \(\alpha_{data}=0.5\)；
- 200 communication rounds；
- local epochs = 3，batch size = 128，learning rate = 0.01。

### SLTrust

- self-anchor \(\alpha=0.5\)；
- EMA coefficient \(\beta=0.1\)；
- recent window \(S=3\)；
- history capacity \(L=10\)。

必须区分 self-anchor \(\alpha\) 与 Dirichlet concentration；正文可把后者记为 \(\alpha_{data}\)。

### Repetition and reporting

- 每个核心配置至少 `[待填：3/5]` 个 seed；
- 报告 mean ± std 或 95% confidence interval；
- 所有图表由 seed 级 JSON 重新生成；
- 说明硬件、PyTorch/CUDA 版本和随机性控制。

## 6.3 Baselines

### Primary baselines

1. **FedAvg/local averaging**：无信誉参考；
2. **SAMA**：与 SLTrust 共享幅度对齐和 self-anchor 的静态相似度基线；
3. **EMA reputation baseline**（代码 `contra`）：SLTrust 的直接机制对照；
4. **EMA + hard cutoff**（代码 `contra_hard`）：检验简单阈值是否足够。

### Secondary localized baselines

- Krum；
- coordinate-wise trimmed mean；
- BALANCE（待核实论文来源和实现对应关系）。

Krum/Trimmed Mean 在 degree=4 的邻域上运行时应标记为 `Local-Krum` / `Local-TrimmedMean`。报告每个诚实节点所见 Byzantine 邻居数分布，并说明原方法理论条件可能不成立。它们可作为经验参考，但不应承担论文主要因果比较；主结论应来自 SLTrust 与 EMA reputation baseline 的同协议比较。

## 6.4 Attacks and Valid Interpretation

### 可用于核心主张

- **Burn-in**：warm-up 后 sign flip；直接测试 reputation inertia；
- **Oscillating**：攻击/休息交替；测试恢复阶段和单向截断。当前实现从第 0 轮直接进入攻击期，并非“先积累信誉再振荡”；若要研究信誉 farming 后的 on–off 利用，应增加初始 warm-up；
- **Sign Flip / Gaussian**：持续明显异常，作为 sanity check；
- **Gradual Flip**：当前实现是对候选模型乘一个从 1 下降到 -1 的标量，中途经过零向量。幅度对齐后，它并不是平滑的小角度旋转，因此只能称 `scalar-ramp sign flip`，不能直接代表隐蔽渐进投毒。

### 当前实现需要谨慎解释

#### Slow Poison

当前代码执行：

```python
return -scale * own_vec
```

该操作仍是完全反向，余弦相似度接近 -1；且后续幅度对齐会消除 `scale` 的幅值效果。因此它**不是“小幅度、保持高相似度的慢投毒”**。当前运行结果只能标记为 `scaled sign flip`，不能用于证明 SLTrust 能识别隐蔽慢投毒。若要形成真正的 slow poison，应另行设计小角度偏移或对 model update 而不是完整参数向量施加渐进扰动。

#### Backdoor

当前代码把固定随机方向叠加到模型更新，没有触发样本、目标标签和 attack success rate。因此它不是标准 backdoor evaluation。当前结果只能标记为 `direction-preserving perturbation proxy`。真正的 backdoor 实验必须：

- 在恶意本地数据中植入触发器；
- 指定目标标签；
- 同时报 clean accuracy 与 ASR；
- 说明 poison ratio 和触发器模式。

#### ALIE

当前 ALIE 使用全部诚实节点的模型更新计算均值和标准差，属于 omniscient central-statistics attacker。该实现与当前 update-based 通信协议一致，但仍需明确攻击者可见全部诚实更新这一强假设。

### Protocol validity checks required before submission

#### No Attack 不是当前意义上的全诚实运行

当前 trainer 即使在 `attack=none` 时仍按 `byzantine_ratio` 划分 Byzantine 节点，而且这些节点不会执行诚实节点的邻居聚合，只保留自己的本地模型。因此当前 `none` 实际表示：

> designated Byzantine nodes send unmodified local updates but do not follow the honest aggregation protocol.

它不能被写成“无 Byzantine 节点的 clean training”。论文至少应补一个 `byzantine_ratio=0` 的真正全诚实实验；当前 `none` 可以保留为 `attack-off` 对照，但标签和解释必须准确。

#### 候选模型协议

当前协议计算：

\[
\cos(v_i^t,\widetilde v_j^t),
\qquad
v_i^t=w_i^{t,+}.
\]

攻击、幅度对齐、信誉更新和邻居聚合都作用于候选模型，最终执行 \(w_i^{t+1}=\alpha v_i^t+(1-\alpha)A_i^t\)。该定义与 SAMA-DFL 的 model-space backbone 一致。旧的累计更新 \(\Delta w\) 协议结果不得与修复后的结果混合。

## 6.5 Metrics

### Utility

- test accuracy；
- test loss；
- consensus error；
- final accuracy 仅作为辅助摘要。

### Post-attack robustness

- 攻击后前 \(H\) 轮平均精度；
- 攻击后最低精度；
- 相对攻击前基线的 accuracy drop；
- recovery rounds。

### Reputation dynamics

- Byzantine-edge mean/median reputation；
- honest-edge mean/median reputation；
- trust separation；
- time-to-neutralize；
- residual trust area；
- ratio 和 \(\varphi\) 的同步曲线。

当前 trainer 只记录 `tau_byz` 平均值，最终论文还需要 honest trust、ratio、phi 和分位数。正在运行的结果可以先用于初步分析，但不足以单独支撑“不会误伤诚实邻居”的结论。

## 6.6 Main Results

### Table 1: Clean and persistent attacks

报告 No Attack、Sign Flip、Gaussian、ALIE。回答：SLTrust 是否保持正常训练效用，以及面对持续攻击是否至少不显著劣于已有方法。

结果填写模板：

> Under no attack, SLTrust changes final accuracy by `[待填]` percentage points relative to the EMA baseline, indicating `[待判断]`. Under persistent attacks, `[只描述有统计支持的结果]`.

### Table 2: Behavior-switching attacks

将 Burn-in、Oscillating、Gradual Flip、BurnIn+scaled-sign-flip 单列，避免主表过宽稀释核心故事。

主要比较：SLTrust vs. EMA reputation baseline / EMA-hard。其他方法为上下文参考。

### Figure 1: Reputation inertia

画 EMA baseline 在 Burn-in 切换前后的：

- Byzantine \(\varphi\)；
- Byzantine \(\tau\)；
- test accuracy。

这张图先证明问题存在，再展示解决方案，不能直接从“ours 更高”开始。

### Figure 2: SLTrust mechanism

同一图中画：

- recent mean；
- background mean；
- ratio；
- SLTrust \(\tau\)。

用竖线标出攻击开始轮。该图验证“为什么有效”，不是只验证“结果更好”。

### Figure 3: Multi-seed utility

以均值线和置信带展示核心攻击下 accuracy；不得只画单 seed 折线。

## 6.7 Ablation Studies

### 当前已有消融

- Full；
- No-ratio；
- Short-only；
- Long-only；
- Median-bg。

其中 Short-only 与 Long-only 都可能近似退化为 ratio=1，解释力有限。当前结果可以保留，但不能声称它们分别严格证明两个窗口各自不可缺。

### 论文定稿前建议补充

- **Overlap-bg**：分母使用包含 recent 的长窗口；验证非重叠设计；
- **No-clip**：不做 `min(1,·)`；验证 oscillating 恢复期为何需要单向截断。

推荐消融问题：

1. ratio 是否必要？Full vs. No-ratio；
2. 非重叠是否必要？Full vs. Overlap-bg；
3. 单向截断是否必要？Full vs. No-clip；
4. 背景统计量是否敏感？Mean-bg vs. Median-bg。

## 6.8 Sensitivity and Generalization

优先级从高到低：

1. \(S,L\)；
2. warm-up length / attack start；
3. topology degree；
4. Dirichlet heterogeneity；
5. Byzantine ratio。

结果讨论不要求所有设置胜出，而要回答：

- 哪些参数决定检测延迟？
- 较短 \(S\) 是否响应快但更易误惩罚诚实波动？
- 较长 \(L\) 是否保留背景更久但增加陈旧性？
- 低 degree 下局部 Byzantine 暴露不均是否导致高方差？

## 6.9 Statistical Analysis

- 预先指定主要指标和核心对比；
- 对多个攻击/方法的成对比较避免只凭视觉判断；
- 报告 effect size 和不确定性，而非只报 p-value；
- 若 seed 较少，使用置信区间和原始散点，避免夸大显著性。

---

# 7. Discussion

## 7.1 What SLTrust Actually Solves

SLTrust 解决的是：

> 当一个已积累信誉的邻居出现可观察的一致性下降时，如何比 EMA 更快撤销其旧信誉。

它不是事前恶意身份识别，也不是对任意 Byzantine 向量的普适检测。

## 7.2 Stability–Responsiveness Trade-off

- 小 \(S\)：更快响应，也更容易把诚实训练波动误判为退化；
- 大 \(S\)：更稳定，但延迟检测；
- 大 \(L\)：历史背景保留更久，但可能不适合真实概念漂移；
- 小 \(\beta\)：EMA 惯性更强，SLTrust 的额外折扣可能更有价值；
- 大 \(\beta\)：EMA 自身已快速响应，SLTrust 的边际收益可能减小。

## 7.3 Limitations

必须明确列出：

1. **可观测性限制**：攻击若保持高 cosine similarity，ratio 不会下降；
2. **诚实非平稳性**：数据漂移和局部震荡也可能触发惩罚；
3. **有限历史**：持续低相似度逐渐填满背景后，ratio 会回到 1；此后依赖已降低的 \(\tau\) 和当前 \(\varphi\)；
4. **局部拓扑依赖**：不同节点看到的 Byzantine 邻居数不一致；
5. **同步假设**：当前实现未处理异步和邻居频繁上下线；
6. **现有 attack proxy**：当前 slow poison 和 backdoor 代码不能等同于标准隐蔽投毒/后门评估；
7. **理论边界**：当前只证明信誉递推性质，不提供完整全局收敛保证。

## 7.4 Broader Interpretation

SLTrust 可被理解为对 EMA 记忆项施加数据驱动的 hazard/forgetting factor。这个视角有助于连接变化检测、自适应滤波和信誉系统，但相关类比必须建立在真实文献上，不能凭直觉声称新颖性。

---

# 8. Conclusion

结论仅重复已经被正文证明或实验支持的内容。建议模板：

> We identified reputation inertia as a concrete weakness of EMA-based neighbor weighting under behavior-switching attacks in decentralized federated learning. SLTrust uses a disjoint recent-to-background comparison to discount inherited reputation only when recent agreement deteriorates. Our analysis establishes conditional faster forgetting without accelerated recovery, while our experiments show `[待填真实结果]`. The method remains limited when malicious updates preserve directional similarity, motivating future work on richer behavioral signals and convergence analysis under history-dependent weights.

不得写：

- “首次从历史感知角度解决”；
- “从根本上切断 farming”；
- “继承 SAMA 鲁棒收敛框架”；
- “对所有攻击均有效”。

---

# 9. Appendix Plan

## Appendix A — Complete Proofs

- reputation range；
- no-amplification proposition；
- conditional forgetting bound；
- conservative non-degradation property。

## Appendix B — Implementation Details

- 模型结构；
- 数据划分；
- 拓扑生成；
- Byzantine 节点选择；
- 参数默认值；
- 零范数与全零 trust 回退；
- 运行环境。

## Appendix C — Full Results

- 每个 seed 的结果；
- 全攻击曲线；
- 完整 sensitivity；
- 负面结果。

## Appendix D — Baseline Applicability

- Local-Krum / Local-TrimmedMean 的局部样本规模；
- 每节点 Byzantine 邻居数分布；
- 原始理论条件是否满足；
- BALANCE 与实现的对应关系。

## Appendix E — Reproducibility

- 配置快照；
- git revision；
- 运行命令；
- seed；
- 原始 JSON schema；
- 生成表图的脚本。

---

# 10. 实验结果返回后的填充顺序

1. 先检查结果是否可复现及多 seed 是否一致；
2. 填 Figure 1，确认 EMA reputation inertia 是否真的存在；
3. 填 Figure 2，确认 SLTrust 的 recent/background/ratio 因果链；
4. 比较 time-to-neutralize 和 residual trust area；
5. 再分析 attack-window accuracy，避免只围绕最终精度讲故事；
6. 检查 No Attack 下诚实节点是否被误惩罚；
7. 最后决定标题、摘要数值和贡献措辞；
8. 若 SLTrust 只在 Burn-in 有优势，收窄为专门防御论文；若跨 behavior-switching 场景稳定，再保留广义标题。

---

# 11. 当前代码与论文证据状态

| 项目 | 当前状态 | 论文处理 |
|---|---|---|
| 非重叠 recent/background | 实现已采用 `hist[:-S]` | 作为正式算法定义 |
| 冷启动 | 背景为空时 ratio=1；首个 tau=phi | 写入 Algorithm 1 |
| 参数约束 | 实现要求 `1 <= S < L` | 写入方法设置 |
| No Attack | 仍保留不参与聚合的 designated Byzantine 节点 | 只能称 attack-off；需补 Byzantine ratio=0 的 clean run |
| 候选模型协议 | 相似度、攻击与聚合均作用于 \(v_i^t=w_i^{t,+}\) | 已切回 SAMA model-space；旧 \(\Delta w\) 结果不得混入新实验 |
| 共同初始化 | 所有客户端复制同一初始模型 | 已修复，保证首轮更新具有共同参考 |
| Burn-in | 已实现 warm-up 后模型更新 sign flip | 核心攻击 |
| Oscillating | 已实现 attack/rest 周期 | 从第 0 轮开始攻击；若主张 farming 后振荡，需增加初始 warm-up |
| Gradual Flip | 标量从 1 过渡到 -1，中途为零 | 只能称 scalar-ramp sign flip，不代表平滑角度投毒 |
| Slow Poison | 实质为 scaled sign flip，且幅度对齐消除 scale | 只能作为补充代理结果，不能证明隐蔽慢投毒能力 |
| Backdoor | 随机方向叠加，无触发数据/ASR | 不能称标准 backdoor，需重实现后再形成安全结论 |
| ALIE | 使用全体诚实向量统计 | 标记 omniscient 设置及与原定义差异 |
| tau 日志 | 仅 Byzantine 平均 tau | 可做初步图；完整论文需 honest/ratio/phi/分位数 |
| 多 seed | 实验进行中 | 等待结果后填统计 |
| 全局收敛 | 尚无独立证明 | 不作为当前贡献 |
| CONTRA 来源 | 当前引用实际是 FoolsGold | 暂称 EMA reputation baseline，待文献核验 |

---

# 12. 一句话审稿人视角检查

论文定稿前，逐条回答：

1. **问题是否真实？** 用 EMA 在切换后的 residual trust 曲线证明，而不是只用直觉。
2. **方法为什么有效？** 用 recent/background/ratio/tau 同步曲线和条件遗忘上界证明。
3. **比较是否公平？** 主要因果比较使用相同协议的 EMA baseline；局部中心化基线只作辅助。
4. **是否过度承诺？** 明确只处理可观察方向下降，保留失败场景。
5. **结果是否可靠？** 多 seed、原始数据、置信区间、配置快照均可复核。
6. **是否真的新？** 文献核验后再下结论，不依赖“首个”措辞。

当这六个问题都有可检查答案时，这个骨架才能转化为一篇可信的论文。
