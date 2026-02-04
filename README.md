# Hyperbolic Petri Net Simulator: Information-Geometric Dynamics

## Core

- **Petri net logic** to model discrete-state transitions
- **Hyperbolic embedding** to represent hierarchical or high-dimensional state spaces
- **Rational inattention constraints** for admissibility verification
- **Dynamic visualization** of trajectories in real time
- **Deterministic fingerprinting** for reproducibility and invariant tracking

## Key Features

- **Canonical Petri Net Simulation**  
  Supports arbitrary incidence matrices and initial markings.

- **Hyperbolic Poincaré Disk Embedding**  
  - Represents system states in a curved, information-geometric space  
  - Preserves hierarchical and exponential growth of state possibilities  
  - Distance metric corresponds to **Fisher Information Distance**  

- **Admissibility and Rational Inattention**  
  - Ensures system transitions respect a bounded hyperbolic displacement  
  - Implements a mathematically rigorous **Rational Inattention constraint**  
  - Detects invariant violations automatically  

- **Dynamic Visualization**  
  - Real-time ASCII and matplotlib plots of radius and hyperbolic distance  
  - Optional PNG output for publication-quality figures  

- **Deterministic Fingerprinting**  
  - SHA3-256 hash of transition trace and final marking  
  - Ensures reproducibility across experiments  

- **Research-Focused Metrics**  
  - Tracks hyperbolic trajectory over steps  
  - Computes admissibility and free-energy-inspired metrics  
  - Supports stochastic and deterministic runs with dynamic seed  


pip install -r requirements.txt   # optional if matplotlib not installed
