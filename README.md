# Graph-JAX Library Source Code Structure
This is a toy Python library transferring Networx onto JAX/XLA
```
......
└── graphjax/                           # Main library package
    ├── pyproject.toml                  # Package configuration
    └── graph_jax/                      # Core library module
        ├── __init__.py                 # Main package initialization
        ├── algorithms/                 # Graph algorithms module
        │   ├── __init__.py             # Algorithms package init
        │   ├── capacity.py             # Capacity dynamics sloving from A. Salgado et al., 2024
        │   ├── pagerank.py             # PageRank algorithm
        │   ├── floyd_warshall.py       # All-pairs shortest paths
        │   ├── algebraic_shortest_path.py # Algebraic shortest path
        │   └── cluster/                # Clustering algorithms
        │       ├── __init__.py         # Cluster package init
        │       ├── kmeans.py           # K-means clustering
        │       ├── spectral.py         # Spectral clustering
        │       ├── soft_clustering.py  # Fuzzy C-means clustering
        │       └── utils.py            # Clustering utilities
        ├── kernels/                    # Computational kernels
        │   ├── __init__.py             # Kernels package init
        │   ├── matrix.py               # Matrix operations
        │   ├── ops.py                  # Graph operations
        │   ├── spgemm.py               # Sparse matrix 
        │   ├── distributed_spgemm.py   # Distributed sparse operations
        │   ├── min_cut.py              # Min-cut algorithms
        │   ├── losses.py               # Loss functions
        │   └── activations.py          # Activation functions
        ├── graphs/                     # Graph data structures
        │   ├── __init__.py             # Graphs package init
        │   ├── graph.py                # Core graph class
        │   ├── io.py                   # Graph I/O operations
        │   └── utils.py                # Graph utilities
        └── utils/                      # Utility functions
            ├── __init__.py             # Utils package init
            ├── hardware_detector.py    # Hardware detection
            └── set_backend.py          # Backend configuration
```

## Dependencies

- **JAX**: Core numerical computing
- **Diffrax**: ODE solving for dynamics
- **NumPy**: Numerical operations
- **NetworkX**: Graph I/O compatibility

## Install
```bash
cd graphjax
pip intall -e .
```
Then feel free to use it like:
```python
import graph_jax as gj
import graph_jax.utils.set_backend
import graph_jax.algorithms.min_plus_shortest_paths
import jax
# please use this config to set up XLA backends, or you can use jax.config()
set_backend('cpu')
```
