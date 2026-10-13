"""
Network topology generation for decentralized FL
"""
import numpy as np
import networkx as nx


def generate_ring_topology(n):
    """
    生成环形拓扑（每个节点连接2个邻居）

    参数:
        n: int - 节点数

    返回:
        List[List[int]] - 邻接列表
    """
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        neighbors[i].append((i - 1) % n)  # 前一个
        neighbors[i].append((i + 1) % n)  # 后一个
    return neighbors


def generate_mesh_topology(n, degree=4, seed=None):
    """生成无向随机正则拓扑，使每个节点严格具有 ``degree`` 个邻居。"""
    if n < 2:
        return [[] for _ in range(n)]
    if not 0 < degree < n:
        raise ValueError(f"degree must satisfy 0 < degree < n, got n={n}, degree={degree}")
    if (n * degree) % 2:
        raise ValueError(f"n * degree must be even, got n={n}, degree={degree}")

    # networkx 使用独立的整数 seed；该整数由调用方设置好的 NumPy RNG
    # 产生，因此同一实验 seed 可复现，同时不依赖 networkx 的全局状态。
    graph_seed = int(np.random.randint(0, 2**32 - 1)) if seed is None else int(seed)
    graph = nx.random_regular_graph(degree, n, seed=graph_seed)
    return [sorted(graph.neighbors(i)) for i in range(n)]


def generate_small_world_topology(n, degree=4, rewire_prob=0.2, seed=None):
    """Generate a connected Watts--Strogatz small-world topology."""
    if not 0 < degree < n or degree % 2:
        raise ValueError(f"small-world degree must be even and satisfy 0 < degree < n, got {degree}")
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError(f"rewire_prob must be in [0, 1], got {rewire_prob}")
    graph_seed = int(np.random.randint(0, 2**32 - 1)) if seed is None else int(seed)
    graph = nx.connected_watts_strogatz_graph(
        n, degree, rewire_prob, tries=100, seed=graph_seed)
    return [sorted(graph.neighbors(i)) for i in range(n)]


def generate_topology(config: dict, num_clients: int, seed: int):
    """Build one configured topology with explicit seed-based reproducibility."""
    topo = config['topology']
    kind = topo['type'].lower()
    if kind == 'ring':
        return generate_ring_topology(num_clients)
    if kind in {'mesh', 'regular'}:
        return generate_mesh_topology(num_clients, degree=topo.get('degree', 4), seed=seed)
    if kind == 'small_world':
        return generate_small_world_topology(
            num_clients, degree=topo.get('degree', 4),
            rewire_prob=topo.get('rewire_prob', 0.2), seed=seed)
    raise ValueError(f"Unknown topology: {kind}")


def generate_erdos_renyi_topology(n, p=0.3):
    """
    生成Erdos-Renyi随机图

    参数:
        n: int - 节点数
        p: float - 边存在概率

    返回:
        List[List[int]] - 邻接列表
    """
    G = nx.erdos_renyi_graph(n, p)
    neighbors = [list(G.neighbors(i)) for i in range(n)]
    return neighbors


def check_connectivity(neighbors, honest_nodes):
    """
    检查诚实节点子图是否连通

    参数:
        neighbors: List[List[int]] - 邻接列表
        honest_nodes: List[int] - 诚实节点ID列表

    返回:
        bool - 是否连通
    """
    honest_set = set(honest_nodes)

    # 构建诚实子图
    G = nx.Graph()
    G.add_nodes_from(honest_nodes)

    for i in honest_nodes:
        for j in neighbors[i]:
            if j in honest_set:
                G.add_edge(i, j)

    return nx.is_connected(G)


def compute_spectral_gap(neighbors, honest_nodes):
    """
    计算诚实节点子图的谱间隙γ

    谱间隙定义: γ = 1 - λ_2(W)
    其中λ_2是混合矩阵W的第二大特征值

    返回:
        float - 谱间隙
    """
    honest_set = set(honest_nodes)
    n_honest = len(honest_nodes)

    # 构建混合矩阵W
    W = np.zeros((n_honest, n_honest))
    node_to_idx = {node: idx for idx, node in enumerate(honest_nodes)}

    for idx, i in enumerate(honest_nodes):
        honest_neighbors = [j for j in neighbors[i] if j in honest_set]
        degree = len(honest_neighbors)

        if degree > 0:
            for j in honest_neighbors:
                j_idx = node_to_idx[j]
                W[idx, j_idx] = 1.0 / (degree + 1)
            W[idx, idx] = 1.0 / (degree + 1)  # 自环权重，保证行和=1

    # 计算特征值
    eigenvalues = np.linalg.eigvals(W)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # 降序

    lambda_1 = eigenvalues[0]  # 应该接近1
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

    gamma = 1 - lambda_2
    return gamma


if __name__ == "__main__":
    # 测试
    print("Testing topology generation...")

    n = 20
    honest = list(range(16))  # 前16个是诚实节点
    byzantine = list(range(16, 20))  # 后4个是拜占庭节点

    # Ring
    ring = generate_ring_topology(n)
    print(f"Ring: Node 0's neighbors = {ring[0]}")
    print(f"Ring: Honest subgraph connected = {check_connectivity(ring, honest)}")
    gamma_ring = compute_spectral_gap(ring, honest)
    print(f"Ring: Spectral gap γ = {gamma_ring:.4f}")

    # Mesh
    mesh = generate_mesh_topology(n, degree=6)
    print(f"\nMesh: Node 0's neighbors = {mesh[0]}")
    print(f"Mesh: Honest subgraph connected = {check_connectivity(mesh, honest)}")
    gamma_mesh = compute_spectral_gap(mesh, honest)
    print(f"Mesh: Spectral gap γ = {gamma_mesh:.4f}")
