import networkx as nx
from graph_jax.graphs import from_networkx, Graph
import jax
jax.config.update('jax_platform_name', 'cpu')

# Create from networkx
nx_g = nx.karate_club_graph()
jax_g = from_networkx(nx_g)

print("Graph created from NetworkX:")
print(f"Number of nodes: {jax_g.n_nodes}")
print(f"Number of edges: {jax_g.n_edges}")
print(f"Graph info: {jax_g}")
print("Adjacency matrix (first 5x5):")
# Modified: call to_adjacency_matrix() method
print(jax_g.to_adjacency_matrix())