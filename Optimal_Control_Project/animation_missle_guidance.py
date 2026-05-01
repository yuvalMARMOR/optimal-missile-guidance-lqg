"""
Optimal Missile Guidance System - LQG Framework
================================================
Animation with Custom Markers

"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

# =============================================================================
# System Parameters
# =============================================================================

class SystemParameters:
    """Container for all system parameters"""
    
    tau = 2.0                    
    V = 914.4                    
    W = 100.0                    
    R_1 = 15e-6                  
    R_2 = 1.67e-3                
    R = 0.0076                   
    t_i = 0.0                    
    t_f = 10.0                   
    dt = 0.01                    
    
    F = np.array([[0, 1, 0],
                  [0, 0, -1],
                  [0, 0, -1/tau]])
    
    B = np.array([[0], [1], [0]])
    G = np.array([[0], [0], [1]])
    
    S_f = np.array([[0.5, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    
    P_0 = np.array([[0, 0, 0],
                    [0, 16, 0],
                    [0, 0, 400]])


# =============================================================================
# Riccati Equation Solver
# =============================================================================

def solve_riccati_equations(params):
    """Solve both Riccati equations with symmetry enforcement"""
    
    n_steps = int((params.t_f - params.t_i) / params.dt)
    time = np.linspace(params.t_i, params.t_f - params.dt, n_steps)
    
    S_list, K_c_list = [], []
    P_list, K_k_list = [], []
    
    # Backward integration for S
    s = params.S_f.copy()
    S_list.append(s.copy())
    
    for t in reversed(time):
        K_c_list.append((1/params.R) * params.B.T @ s)
        S_dot = -(params.F.T @ s + s @ params.F - 
                  s @ params.B @ params.B.T @ s / params.R)
        s = s - S_dot * params.dt
        s = 0.5 * (s + s.T)
        S_list.append(s.copy())
    
    # Forward integration for P
    p = params.P_0.copy()
    P_list.append(p.copy())
    
    for t in time:
        time_to_go = max(params.t_f - t, 1e-6)
        H = np.array([[1/(params.V * time_to_go), 0, 0]])
        M = params.R_1 + params.R_2 / (time_to_go**2)
        
        K_k_list.append(p @ H.T / M)
        P_dot = (params.F @ p + p @ params.F.T - 
                 p @ H.T @ H @ p / M + 
                 params.G @ params.G.T * params.W)
        p = p + P_dot * params.dt
        p = 0.5 * (p + p.T)
        P_list.append(p.copy())
    
    K_c = np.flip(np.array(K_c_list), axis=0)
    S = np.flip(np.array(S_list)[:-1], axis=0)
    K_k = np.array(K_k_list)
    P = np.array(P_list)[:-1]
    
    return K_c, K_k, P, S, time


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(params, K_c, K_k, time):
    """Run closed-loop simulation"""
    
    x = np.array([[np.random.normal(0, np.sqrt(params.P_0[i,i]))] 
                  for i in range(3)])
    x_hat = np.zeros((3, 1))
    
    X, X_hat, U = [], [], []
    
    for j, t in enumerate(time):
        X.append(x.copy())
        X_hat.append(x_hat.copy())
        
        time_to_go = max(params.t_f - t, 1e-6)
        H = np.array([[1/(params.V * time_to_go), 0, 0]])
        M = params.R_1 + params.R_2 / (time_to_go**2)
        
        w = np.random.normal(0, np.sqrt(params.W / params.dt))
        m = np.random.normal(0, np.sqrt(M / params.dt))
        
        u = -K_c[j] @ x_hat
        U.append(u.copy())
        
        x_dot = params.F @ x + params.B @ u + params.G * w
        x_hat_dot = (params.F @ x_hat + params.B @ u + 
                     K_k[j] * (H @ x - H @ x_hat + m))
        
        x = x + x_dot * params.dt
        x_hat = x_hat + x_hat_dot * params.dt
    
    return np.array(X), np.array(X_hat), np.array(U)


def create_target_trajectory_from_simulation(X, time, params):
    """
    Generate target trajectory from simulated a_T.
    This ensures the animation is consistent with the actual simulation.
    """
    a_T = X[:, 2, 0]  # Extract a_T from simulation results
    
    v_t, y_t = 0, 0
    Y_T = []
    
    for k in range(len(time)):
        v_t = v_t + a_T[k] * params.dt
        y_t = y_t + v_t * params.dt
        Y_T.append(y_t)
    
    return np.array(Y_T)


# =============================================================================
# Animation
# =============================================================================

def create_animation(time, X, X_hat, Y_T, params, 
                     save_gif=True, filename='missile_guidance.gif'):
    """Create animation with custom markers"""
    
    # Extract trajectories
    Y_pursuer_rel = X[:, 0, 0]
    Y_pursuer_est_rel = X_hat[:, 0, 0]
    Y_pursuer = Y_pursuer_rel + Y_T
    Y_pursuer_est = Y_pursuer_est_rel + Y_T
    X_pursuer = params.V * time
    X_target = np.ones_like(time) * params.V * params.t_f
    
    # Setup figure - dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    # Axis limits
    padding = 200
    ax.set_xlim(-padding, params.V * params.t_f + padding)
    y_range = max(np.max(np.abs(Y_pursuer)), np.max(np.abs(Y_T))) + 150
    ax.set_ylim(-y_range, y_range)
    
    # Grid
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    ax.set_axisbelow(True)
    
    # ===== MARKERS =====
    
    # Missile (Actual) - DIAMOND shape
    missile_marker, = ax.plot([], [], 'D', color='#00d4ff', markersize=16,
                               markeredgecolor='white', markeredgewidth=2,
                               label='Missile (Actual)', zorder=10)
    missile_glow, = ax.plot([], [], 'D', color='#00d4ff', markersize=28,
                             alpha=0.3, zorder=9)
    
    # Missile (Estimated) - TRIANGLE shape
    missile_est, = ax.plot([], [], '^', color='#7b68ee', markersize=14,
                            markeredgecolor='white', markeredgewidth=1.5,
                            alpha=0.8, label='Missile (Estimated)', zorder=8)
    
    # Target - X shape
    target_marker, = ax.plot([], [], 'X', color='#ff4757', markersize=22,
                              markeredgecolor='white', markeredgewidth=2,
                              label='Target', zorder=10)
    target_glow, = ax.plot([], [], 'o', color='#ff4757', markersize=35,
                            alpha=0.2, zorder=9)
    
    # Trails
    missile_trail, = ax.plot([], [], '-', color='#00d4ff', alpha=0.6, 
                              linewidth=3, zorder=5)
    target_trail, = ax.plot([], [], '-', color='#ff4757', alpha=0.6, 
                             linewidth=3, zorder=5)
    estimate_trail, = ax.plot([], [], '--', color='#7b68ee', alpha=0.4, 
                               linewidth=2, zorder=4)
    
    # Line of Sight
    los_line, = ax.plot([], [], ':', color='#ffd700', alpha=0.5, 
                         linewidth=2, label='Line of Sight', zorder=3)
    
    # Info panel
    info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f3460',
                                edgecolor='#00d4ff', alpha=0.9),
                       color='white')
    
    # Miss distance display
    miss_box = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=14,
                       verticalalignment='bottom', fontweight='bold',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460',
                                edgecolor='white', alpha=0.9),
                       color='white')
    
    # Progress bar
    progress_bg = plt.Rectangle((0.3, 0.02), 0.4, 0.03, transform=ax.transAxes,
                                  facecolor='#0f3460', edgecolor='white',
                                  alpha=0.8, zorder=20)
    ax.add_patch(progress_bg)
    
    progress_bar = plt.Rectangle((0.3, 0.02), 0, 0.03, transform=ax.transAxes,
                                   facecolor='#00d4ff', alpha=0.8, zorder=21)
    ax.add_patch(progress_bar)
    
    progress_text = ax.text(0.5, 0.035, '', transform=ax.transAxes, fontsize=10,
                            ha='center', va='center', color='white', 
                            fontweight='bold', fontfamily='monospace', zorder=22)
    
    # Labels
    ax.set_xlabel('Distance [m]', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Cross-Range [m]', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Optimal Missile Guidance System (LQG Control)', 
                 fontsize=18, color='white', fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=11, facecolor='#0f3460',
                       edgecolor='#00d4ff', labelcolor='white')
    legend.get_frame().set_alpha(0.9)
    
    # Styling
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#00d4ff')
        spine.set_linewidth(2)
    
    def animate(frame):
        # Trail
        trail_len = min(100, frame + 1)
        start = max(0, frame - trail_len + 1)
        
        # Update positions
        missile_marker.set_data([X_pursuer[frame]], [Y_pursuer[frame]])
        missile_glow.set_data([X_pursuer[frame]], [Y_pursuer[frame]])
        missile_est.set_data([X_pursuer[frame]], [Y_pursuer_est[frame]])
        target_marker.set_data([X_target[frame]], [Y_T[frame]])
        target_glow.set_data([X_target[frame]], [Y_T[frame]])
        
        # Update trails
        missile_trail.set_data(X_pursuer[start:frame+1], Y_pursuer[start:frame+1])
        target_trail.set_data(X_target[start:frame+1], Y_T[start:frame+1])
        estimate_trail.set_data(X_pursuer[start:frame+1], Y_pursuer_est[start:frame+1])
        
        # LOS
        los_line.set_data([X_pursuer[frame], X_target[frame]], 
                          [Y_pursuer[frame], Y_T[frame]])
        
        # Metrics
        t_current = time[frame]
        t_remaining = params.t_f - t_current
        miss_dist = Y_pursuer_rel[frame]
        range_to_target = np.sqrt((X_target[frame] - X_pursuer[frame])**2 + 
                                   (Y_T[frame] - Y_pursuer[frame])**2)
        
        # Info text
        info_box.set_text(
            f'TIME: {t_current:6.2f} s\n'
            f'T-GO: {t_remaining:6.2f} s\n'
            f'--------------------\n'
            f'RANGE: {range_to_target:8.0f} m\n'
            f'SPEED: {params.V:8.1f} m/s\n'
            f'--------------------\n'
            f'MISSILE Y: {Y_pursuer[frame]:+8.1f} m\n'
            f'TARGET Y:  {Y_T[frame]:+8.1f} m'
        )
        
        # Miss distance color
        if abs(miss_dist) < 10:
            miss_color = '#2ed573'
            status = 'LOCKED'
        elif abs(miss_dist) < 50:
            miss_color = '#ffa502'
            status = 'TRACKING'
        else:
            miss_color = '#ff4757'
            status = 'ACQUIRING'
        
        miss_box.set_text(f'{status}  |  Miss: {miss_dist:+.2f} m')
        miss_box.set_bbox(dict(boxstyle='round,pad=0.5', facecolor=miss_color,
                               edgecolor='white', alpha=0.9))
        miss_box.set_color('white' if abs(miss_dist) >= 10 else 'black')
        
        # Progress
        progress = (frame + 1) / len(time)
        progress_bar.set_width(0.4 * progress)
        progress_text.set_text(f'{progress*100:.0f}%')
        
        return (missile_marker, missile_glow, missile_est, target_marker, 
                target_glow, missile_trail, target_trail, estimate_trail,
                los_line, info_box, miss_box, progress_bar, progress_text)
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                   interval=30, blit=True, repeat=True)
    
    plt.tight_layout()
    
    if save_gif:
        print(f"Saving animation as {filename}...")
        writer = animation.PillowWriter(fps=30)
        anim.save(filename, writer=writer, dpi=100)
        print("Animation saved!")
    
    plt.show()
    
    return anim


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("   LQG MISSILE GUIDANCE SYSTEM")
    print("=" * 60)
    
    params = SystemParameters()
    
    print("\n[1/4] Solving Riccati equations...")
    K_c, K_k, P, S, time = solve_riccati_equations(params)
    print(f"      Done - {len(time)} steps")
    
    print("\n[2/4] Running simulation...")
    X, X_hat, U = run_simulation(params, K_c, K_k, time)
    print(f"      Done - Final miss: {X[-1, 0, 0]:.2f} m")
    
    print("\n[3/4] Generating target trajectory (from simulation)...")
    Y_T = create_target_trajectory_from_simulation(X, time, params)
    print(f"      Done")
    
    print("\n[4/4] Creating animation...")
    anim = create_animation(time, X, X_hat, Y_T, params,
                            save_gif=True,
                            filename='missile_guidance.gif')
    
    print("\n" + "=" * 60)
    print("   COMPLETE")
    print("=" * 60)
    
    return anim


if __name__ == "__main__":
    main()