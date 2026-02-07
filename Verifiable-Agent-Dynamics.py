import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Final, Optional

# =====================================================================
# 1. PARAMETRIC SCHEMA
# =====================================================================

@dataclass(frozen=True)
class PhysicsConfig:
    """Production-grade constants for Riemannian-Langevin flow."""
    RELAXATION_GAMMA: float = 0.88    
    THERMAL_SIGMA: float = 0.035      
    TIME_STEP_DT: float = 0.14        
    CURVATURE_LAMBDA: float = 0.30    
    GRID_DENSITY: int = 120           
    N_ENSEMBLE: int = 1500            
    GEODESIC_SAMPLES: int = 85        
    FPS: int = 30                     
    LIMIT: float = 9.0                
    SEED: Optional[int] = None

# =====================================================================
# 2. COMPUTATIONAL CORE
# =====================================================================

class RiemannianDynamicalSystem:
    def __init__(self, cfg: PhysicsConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.SEED if cfg.SEED else secrets.randbits(32))
        
        self.q = self.rng.standard_normal((cfg.N_ENSEMBLE, 2)) * 3.5
        self.p = np.zeros((cfg.N_ENSEMBLE, 2))
        self.phi_source = np.array([0.0, 0.0])
        
        self.local_g_inv = np.ones(cfg.N_ENSEMBLE)
        self.epoch, self.ledger_hash = 0, ""
        self.kl_div, self.entropy, self.tension = 0.0, 0.0, 0.0
        self._generate_hash()

    def update_physics(self):
        t = time.time()
        self.phi_source = np.array([3.4 * np.sin(0.35 * t), 3.4 * np.cos(0.45 * t + 0.5)])
        
        dq = self.phi_source - self.q
        r_sq = np.einsum('ij,ij->i', dq, dq)
        
        # G^-1(q): Fisher Information Metric Inverse
        self.local_g_inv = 1.0 / (1.0 + self.cfg.CURVATURE_LAMBDA * r_sq)
        natural_gradient = dq * self.local_g_inv[:, np.newaxis]
        
        noise = self.rng.standard_normal((self.cfg.N_ENSEMBLE, 2)) * self.cfg.THERMAL_SIGMA
        self.p = (self.cfg.RELAXATION_GAMMA * self.p + natural_gradient + noise)
        self.q += self.p * self.cfg.TIME_STEP_DT

        self.kl_div = float(np.mean(0.5 * r_sq))
        self.tension = float(np.mean(1.0 - self.local_g_inv))
        var = float(np.var(self.q, axis=0).mean())
        self.entropy = 0.5 * np.log(2 * np.pi * np.e * (var + 1e-9))
        
        self.epoch += 1
        if self.epoch % 20 == 0: self._generate_hash()

    def _generate_hash(self):
        self.ledger_hash = hashlib.sha256(self.q.tobytes()).hexdigest()

# =====================================================================
# 3. PRODUCTION DASHBOARD
# =====================================================================

class ResearchDashboard:
    def __init__(self, sys: RiemannianDynamicalSystem):
        self.sys = sys
        self.fig = plt.figure(figsize=(19, 10), facecolor='#000000')
        gs = self.fig.add_gridspec(1, 2, wspace=0.01)
        self.ax_m = self.fig.add_subplot(gs[0, 0], facecolor='#000000') 
        self.ax_p = self.fig.add_subplot(gs[0, 1], facecolor='#000000')
        
        self._init_artists()
        self._init_hud()

    def _init_artists(self):
        grid = np.linspace(-self.sys.cfg.LIMIT, self.sys.cfg.LIMIT, self.sys.cfg.GRID_DENSITY)
        self.gx, self.gy = np.meshgrid(grid, grid)
        self.grid_stack = np.stack([self.gx.ravel(), self.gy.ravel()], axis=1)
        
        # Manifold Heatmap
        self.phi_mesh = self.ax_m.pcolormesh(self.gx, self.gy, np.zeros_like(self.gx), 
                                             shading='gouraud', cmap='magma', norm=Normalize(0, 1))
        
        # Multi-Variable Phase Scatter
        self.ensemble = self.ax_p.scatter(self.sys.q[:, 0], self.sys.q[:, 1], 
                                          c=np.zeros(self.sys.cfg.N_ENSEMBLE), 
                                          s=20, edgecolors='none', alpha=0.7, 
                                          cmap='magma', norm=Normalize(0, 0.6))
        
        self.geodesics = LineCollection([], colors='#00FF9C', alpha=0.15, linewidths=1.2)
        self.ax_p.add_collection(self.geodesics)
        
        for ax in [self.ax_m, self.ax_p]:
            ax.set_xlim(-self.sys.cfg.LIMIT, self.sys.cfg.LIMIT)
            ax.set_ylim(-self.sys.cfg.LIMIT, self.sys.cfg.LIMIT)
            ax.set_xticks([]); ax.set_yticks([])

    def _init_hud(self):
        self.head = self.fig.text(0.5, 0.95, "", ha='center', color='#00FF9C', fontfamily='monospace', fontweight='bold',
                                  bbox=dict(facecolor='#001A11', edgecolor='#00FF9C', alpha=0.9, pad=5))
        self.foot = self.fig.text(0.5, 0.04, "", ha='center', color='white', fontfamily='monospace', fontsize=10)
        
        label_style = dict(color='white', fontsize=9, fontfamily='monospace', 
                           bbox=dict(facecolor='#000000', edgecolor='#333333', alpha=0.8, pad=10))
        
        # FULLY LABELED KEYMAPS
        self.fig.text(0.06, 0.06, 
            "RIEMANNIAN STATE [M]\n"
            "----------------------------\n"
            "● BLACK/PURPLE : Low Potential Φ\n"
            "● ORANGE/WHITE : High Info Density\n"
            "● SMOOTH GRAD  : Fisher Metric Field\n"
            "● XY PLANE     : Coordinate Space (q)", **label_style)

        self.fig.text(0.74, 0.06, 
            "PHASE SPACE [P]\n"
            "----------------------------\n"
            "● MARKER COLOR : Kinetic Energy (p²)\n"
            "  (Dark: Static | Bright: High Velocity)\n"
            "● MARKER SIZE  : Local Metric Tension\n"
            "● GREEN TRACE  : Geodesic Projection\n"
            "● DENSITY      : Probability Flow", **label_style)

    def update(self, frame):
        self.sys.update_physics()
        
        # Vectorized Manifold Refresh
        dist_sq = np.sum((self.grid_stack - self.sys.phi_source)**2, axis=1)
        phi = np.exp(-0.35 * dist_sq).reshape(self.gx.shape)
        self.phi_mesh.set_array(phi.ravel())
        
        # Physics Mapping
        p_energy = np.linalg.norm(self.sys.p, axis=1)
        tension_size = (1.0 - self.sys.local_g_inv) * 150.0 + 10.0
        
        self.ensemble.set_offsets(self.sys.q)
        self.ensemble.set_array(p_energy)
        self.ensemble.set_sizes(tension_size)
        
        idx = np.linspace(0, self.sys.cfg.N_ENSEMBLE-1, self.sys.cfg.GEODESIC_SAMPLES, dtype=int)
        self.geodesics.set_segments(np.stack([self.sys.q[idx], np.tile(self.sys.phi_source, (len(idx), 1))], axis=1))

        self.head.set_text(f"LEDGER HASH: {self.sys.ledger_hash.upper()}")
        self.foot.set_text(f"EPOCH: {self.sys.epoch:04d}  |  KL DIV: {self.sys.kl_div:.5f}  |  "
                           f"ENTROPY: {self.sys.entropy:.5f}  |  TENSION: {self.sys.tension:.5f}")
        return []

if __name__ == "__main__":
    sys_core = RiemannianDynamicalSystem(PhysicsConfig())
    dash_ui = ResearchDashboard(sys_core)
    ani = FuncAnimation(dash_ui.fig, dash_ui.update, interval=1000//sys_core.cfg.FPS, 
                        blit=False, cache_frame_data=False)
    plt.show()
