import networkx as nx
from graph_jax.graphs import from_networkx, Graph
import jax
jax.config.update('jax_platform_name', 'cpu')

# 从 networkx 创建
nx_g = nx.karate_club_graph()
jax_g = from_networkx(nx_g)

print("从 NetworkX 创建的图:")
print(f"节点数: {jax_g.n_nodes}")
print(f"边数: {jax_g.n_edges}")
print(f"图信息: {jax_g}")
print("邻接矩阵 (前5x5):")
# 修改: 调用 to_adjacency_matrix() 方法
print(jax_g.to_adjacency_matrix())