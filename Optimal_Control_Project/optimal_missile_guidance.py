"""
Optimal Missile Guidance using LQG Control
==========================================
This module implements an optimal missile guidance system using 
Linear Quadratic Gaussian (LQG) control theory.

The objective is to minimize the final miss distance while using 
minimal control effort (lateral acceleration).



"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# =============================================================================
# Graph Style Configuration
# =============================================================================

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Use Dark2 color palette
COLORS = plt.cm.Dark2.colors

# Global plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.figsize': (10, 6),
    'axes.prop_cycle': plt.cycler(color=COLORS)
})


# =============================================================================
# Configuration and Parameters
# =============================================================================

@dataclass
class SystemParameters:
    """Physical and control parameters for the missile guidance system."""
    
    # Time parameters
    tau: float = 2.0            # Target correlation time [sec]
    t_initial: float = 0.0      # Initial time [sec]
    t_final: float = 10.0       # Final time [sec]
    dt: float = 0.01            # Time step [sec]
    
    # Physical parameters
    V: float = 914.4            # Closing velocity [m/s]
    W: float = 100.0            # Process noise intensity [m²/s⁵]
    
    # Measurement noise parameters
    R1: float = 15e-6           # Base measurement noise [rad²/s²]
    R2: float = 1.67e-3         # Time-varying noise component [rad²/s³]
    
    # Control weight - FIXED!
    b: float = 0.0152           # Control weighting factor from problem statement
    R: float = 0.0076           # R = b/2 for the quadratic cost function
    
    def __post_init__(self):
        """Initialize system matrices after dataclass creation."""
        # Ensure R = b/2 (in case b is modified)
        self.R = self.b / 2
        self._setup_matrices()
    
    def _setup_matrices(self):
        """Set up state-space matrices."""
        # State transition matrix
        self.F = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, -1/self.tau]
        ])
        
        # Control input matrix
        self.B = np.array([[0], [1], [0]])
        
        # Process noise matrix
        self.G = np.array([[0], [0], [1]])
        
        # Terminal cost matrix (penalizes final position error)
        self.S_final = np.array([
            [0.5, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        # Initial covariance matrix
        self.P_initial = np.array([
            [0, 0, 0],
            [0, 16, 0],
            [0, 0, 400]
        ])
    
    def get_measurement_matrix(self, time_to_go: float) -> np.ndarray:
        """Calculate time-varying measurement matrix H(t)."""
        return np.array([[1 / (self.V * time_to_go), 0, 0]])
    
    def get_measurement_noise(self, time_to_go: float) -> float:
        """Calculate time-varying measurement noise intensity M(t)."""
        return self.R1 + self.R2 / (time_to_go ** 2)


# =============================================================================
# Riccati Equation Solvers
# =============================================================================

class RiccatiSolver:
    """Solver for the control and estimation Riccati equations."""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.time_array = self._create_time_array()
    
    def _create_time_array(self) -> np.ndarray:
        """Create time array excluding final time to avoid division by zero."""
        n_steps = int((self.params.t_final - self.params.t_initial) / self.params.dt)
        return np.linspace(self.params.t_initial, 
                          self.params.t_final - self.params.dt, 
                          n_steps)
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve both Riccati equations.
        
        Returns:
            K_control: Control gain matrix over time
            K_kalman: Kalman gain matrix over time
            S: Control Riccati solution
            P: Estimation Riccati solution (covariance)
        """
        S, K_control = self._solve_control_riccati()
        P, K_kalman = self._solve_estimation_riccati()
        
        self._print_array_sizes(K_control, K_kalman, S, P)
        
        return K_control, K_kalman, S, P
    
    def _solve_control_riccati(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the control Riccati equation backward in time.
        
        Equation: -Ṡ = FᵀS + SF - SBR⁻¹BᵀS + Q
        """
        p = self.params
        S_list = [p.S_final.copy()]
        K_control_list = []
        
        S_current = p.S_final.copy()
        
        for _ in reversed(self.time_array):
            # Compute control gain: Kc = R⁻¹BᵀS
            K_c = (1 / p.R) * p.B.T @ S_current
            K_control_list.append(K_c.copy())
            
            # Riccati equation derivative
            S_dot = -(p.F.T @ S_current + 
                     S_current @ p.F - 
                     S_current @ p.B @ p.B.T @ S_current / p.R)
            
            # Euler integration (backward)
            S_current = S_current - S_dot * p.dt
            S_list.append(S_current.copy())
        
        # Reverse to match forward time order
        S_array = np.flip(np.array(S_list)[:-1], axis=0)
        K_control_array = np.flip(np.array(K_control_list), axis=0)
        
        return S_array, K_control_array
    
    def _solve_estimation_riccati(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the estimation Riccati equation forward in time.
        
        Equation: Ṗ = FP + PFᵀ - PHᵀM⁻¹HP + GWGᵀ
        """
        p = self.params
        P_list = [p.P_initial.copy()]
        K_kalman_list = []
        
        P_current = p.P_initial.copy()
        
        for t in self.time_array:
            time_to_go = max(p.t_final - t, 1e-6)
            
            H = p.get_measurement_matrix(time_to_go)
            M = p.get_measurement_noise(time_to_go)
            
            # Compute Kalman gain: Kk = PHᵀM⁻¹
            K_k = P_current @ H.T / M
            K_kalman_list.append(K_k.copy())
            
            # Riccati equation derivative
            P_dot = (p.F @ P_current + 
                    P_current @ p.F.T - 
                    P_current @ H.T @ H @ P_current / M + 
                    p.G @ p.G.T * p.W)
            
            # Euler integration (forward)
            P_current = P_current + P_dot * p.dt
            P_list.append(P_current.copy())
        
        P_array = np.array(P_list)[:-1]
        K_kalman_array = np.array(K_kalman_list)
        
        return P_array, K_kalman_array
    
    def _print_array_sizes(self, K_c, K_k, S, P):
        """Print dimensions of computed arrays for verification."""
        print(f"\nFinal array sizes:")
        print(f"  time: {len(self.time_array)}")
        print(f"  K_c: {K_c.shape}")
        print(f"  K_k: {K_k.shape}")
        print(f"  S: {S.shape}")
        print(f"  P: {P.shape}")


# =============================================================================
# Simulation Engine
# =============================================================================

class GuidanceSimulator:
    """Simulates the missile guidance system with optimal control."""
    
    def __init__(self, params: SystemParameters, time_array: np.ndarray):
        self.params = params
        self.time = time_array
    
    def run(self, K_control: np.ndarray, K_kalman: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the guidance simulation.
        
        Args:
            K_control: Control gain matrix over time
            K_kalman: Kalman gain matrix over time
            
        Returns:
            X: True state trajectory
            X_hat: Estimated state trajectory
            U: Control input history
        """
        # Initialize states
        x, x_hat = self._initialize_states()
        
        # Pre-generate noise sequences
        measurement_noise, process_noise = self._generate_noise()
        
        # Storage for trajectories
        X = [x.copy()]
        X_hat = [x_hat.copy()]
        U = []
        
        # Simulation loop
        for k, t in enumerate(self.time):
            time_to_go = self.params.t_final - t
            H = self.params.get_measurement_matrix(time_to_go)
            
            # Optimal control law: u = -Kc * x̂
            u = -K_control[k] @ x_hat
            U.append(u.copy())
            
            # State dynamics: ẋ = Fx + Bu + Gw
            x_dot = (self.params.F @ x + 
                    self.params.B @ u + 
                    self.params.G * process_noise[k])
            
            # Estimator dynamics with Kalman correction
            innovation = H @ x - H @ x_hat + measurement_noise[k]
            x_hat_dot = (self.params.F @ x_hat + 
                        self.params.B @ u + 
                        K_kalman[k] * innovation)
            
            # Euler integration
            x = x + x_dot * self.params.dt
            x_hat = x_hat + x_hat_dot * self.params.dt
            
            # Store (except for last iteration to match time array size)
            if k < len(self.time) - 1:
                X.append(x.copy())
                X_hat.append(x_hat.copy())
        
        return np.array(X), np.array(X_hat), np.array(U)
    
    def _initialize_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize true and estimated states."""
        P0 = self.params.P_initial
        
        # True initial state: random according to initial covariance
        x0 = np.array([[np.random.normal(0, np.sqrt(P0[0, 0]))],
                       [np.random.normal(0, np.sqrt(P0[1, 1]))],
                       [np.random.normal(0, np.sqrt(P0[2, 2]))]])
        
        # Estimated initial state: zero (no prior information)
        x_hat0 = np.zeros((3, 1))
        
        return x0, x_hat0
    
    def _generate_noise(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-generate measurement and process noise sequences."""
        p = self.params
        
        measurement_noise = []
        process_noise = []
        
        for t in self.time:
            time_to_go = p.t_final - t
            M = p.get_measurement_noise(time_to_go)
            
            # Scale noise by sqrt(intensity/dt) for discrete simulation
            m = np.random.normal(0, np.sqrt(M / p.dt))
            w = np.random.normal(0, np.sqrt(p.W / p.dt))
            
            measurement_noise.append(m)
            process_noise.append(w)
        
        return np.array(measurement_noise), np.array(process_noise)


# =============================================================================
# Cost Function Calculator
# =============================================================================

class CostCalculator:
    """Calculates the optimal cost-to-go function."""
    
    def __init__(self, params: SystemParameters, time_array: np.ndarray):
        self.params = params
        self.time = time_array
    
    def compute_cost_to_go(self, X_hat: np.ndarray, S: np.ndarray,
                           K_kalman: np.ndarray, P: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                    np.ndarray, np.ndarray]:
        """
        Compute the optimal cost-to-go function.
        
        J° = x̂ᵀSx̂ + ∫tr(KMKᵀS)dτ + tr(P_f·S_f)
        
        Returns:
            a: State cost component (x̂ᵀSx̂)
            b: Integrand of control effort term
            c: Terminal cost component
            I: Integral of control effort
            J: Total cost-to-go
        """
        a = []  # x̂ᵀSx̂
        b = []  # tr(KMKᵀS)
        c = []  # tr(P_f·S_f)
        
        # Terminal cost (constant)
        terminal_cost = np.trace(P[-1] @ self.params.S_final)
        
        for k, t in enumerate(self.time):
            time_to_go = self.params.t_final - t
            M = self.params.get_measurement_noise(time_to_go)
            
            # State cost: x̂ᵀSx̂
            state_cost = X_hat[k].T @ S[k] @ X_hat[k]
            a.append(state_cost)
            
            # Control effort integrand: tr(KMKᵀS)
            effort_integrand = np.trace(K_kalman[k] @ K_kalman[k].T @ S[k] * M)
            b.append(effort_integrand)
            
            # Terminal cost (same for all time steps)
            c.append([[terminal_cost]])
        
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Compute integral from t to t_f
        I = self._compute_integral(b)
        
        # Total cost-to-go
        J = I + a + c
        
        return a, b, c, I, J
    
    def _compute_integral(self, integrand: np.ndarray) -> np.ndarray:
        """Compute the integral ∫_t^tf b(τ)dτ for each time t."""
        integral = []
        
        for k in range(len(self.time)):
            val = np.trapz(integrand[k:], self.time[k:], self.params.dt)
            integral.append([[val]])
        
        return np.array(integral)
    
    def compute_final_cost(self, X: np.ndarray, U: np.ndarray) -> float:
        """Compute the actual final cost from simulation results."""
        terminal_cost = 0.5 * X[-1, 0, 0] ** 2
        control_cost = 0.5 * self.params.b * np.trapz(U[:,0,0]**2, self.time, self.params.dt)
        return terminal_cost + control_cost


# =============================================================================
# Visualization
# =============================================================================

class ResultsPlotter:
    """Plots simulation results and analysis."""
    
    def __init__(self, time_array: np.ndarray):
        self.time = time_array
        self.colors = COLORS
    
    def plot_control_gains(self, K_control: np.ndarray):
        """Plot control gains vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, K_control[:, 0, 0], color=self.colors[0], label=r'$K_{c1}$ [1/s²]')
        ax.plot(self.time, K_control[:, 0, 1], color=self.colors[1], label=r'$K_{c2}$ [1/s]')
        ax.plot(self.time, K_control[:, 0, 2], color=self.colors[2], label=r'$K_{c3}$ [-]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Control Gains')
        ax.set_title('Time-Varying Control Gains')
        ax.legend()
        plt.tight_layout()
        plt.savefig('control_gains.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_kalman_gains(self, K_kalman: np.ndarray):
        """Plot Kalman gains vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, K_kalman[:, 0, 0], color=self.colors[0], label=r'$K_{k1}$ [m]')
        ax.plot(self.time, K_kalman[:, 1, 0], color=self.colors[1], label=r'$K_{k2}$ [m/s]')
        ax.plot(self.time, K_kalman[:, 2, 0], color=self.colors[2], label=r'$K_{k3}$ [m/s²]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Kalman Gains')
        ax.set_title('Time-Varying Kalman Gains')
        ax.legend()
        plt.tight_layout()
        plt.savefig('kalman_gains.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_covariance(self, P: np.ndarray):
        """Plot covariance matrix elements vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, P[:, 0, 0], color=self.colors[0], label=r'$P_{11}$ [m²]')
        ax.plot(self.time, P[:, 1, 0], color=self.colors[1], label=r'$P_{21}$ [m·(m/s)]')
        ax.plot(self.time, P[:, 2, 0], color=self.colors[2], label=r'$P_{31}$ [m·(m/s²)]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Variance/Covariance')
        ax.set_title('Filter Covariance (First Column)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('covariance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_full_covariance(self, P: np.ndarray):
        """Plot all main covariance elements vs time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.time, P[:, 0, 0], color=self.colors[0], label=r'$P_{11}$')
        ax.plot(self.time, P[:, 1, 1], color=self.colors[1], label=r'$P_{22}$')
        ax.plot(self.time, P[:, 2, 2], color=self.colors[2], label=r'$P_{33}$')
        ax.plot(self.time, P[:, 1, 0], color=self.colors[3], label=r'$P_{21}$')
        ax.plot(self.time, P[:, 2, 0], color=self.colors[4], label=r'$P_{31}$')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Variance/Covariance')
        ax.set_title('Full Covariance Matrix Elements')
        ax.legend()
        plt.tight_layout()
        plt.savefig('full_covariance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_states(self, X: np.ndarray, X_hat: np.ndarray):
        """Plot true and estimated state trajectories."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Position
        axes[0].plot(self.time, X[:, 0, 0], color=self.colors[0], label=r'$y$ (True)')
        axes[0].plot(self.time, X_hat[:, 0, 0], color=self.colors[1], linestyle='--', label=r'$\hat{y}$ (Estimated)')
        axes[0].set_ylabel('y [m]')
        axes[0].set_title('Position')
        axes[0].legend()
        
        # Velocity
        axes[1].plot(self.time, X[:, 1, 0], color=self.colors[0], label=r'$v$ (True)')
        axes[1].plot(self.time, X_hat[:, 1, 0], color=self.colors[1], linestyle='--', label=r'$\hat{v}$ (Estimated)')
        axes[1].set_ylabel('v [m/s]')
        axes[1].set_title('Velocity')
        axes[1].legend()
        
        # Acceleration
        axes[2].plot(self.time, X[:, 2, 0], color=self.colors[0], label=r'$a_T$ (True)')
        axes[2].plot(self.time, X_hat[:, 2, 0], color=self.colors[1], linestyle='--', label=r'$\hat{a}_T$ (Estimated)')
        axes[2].set_xlabel('Time [sec]')
        axes[2].set_ylabel(r'$a_T$ [m/s²]')
        axes[2].set_title('Target Acceleration')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('states.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_control_input(self, U: np.ndarray):
        """Plot optimal control input vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, U[:, 0, 0], color=self.colors[0])
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel(r'$u^0(t)$ [m/s²]')
        ax.set_title('Optimal Control Input')
        plt.tight_layout()
        plt.savefig('control_input.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_cost_to_go(self, J: np.ndarray):
        """Plot total cost-to-go vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, J[:, 0, 0], color=self.colors[0])
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Optimal Cost-to-go')
        ax.set_title('Cost-to-Go Function')
        plt.tight_layout()
        plt.savefig('cost_to_go.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_cost_components(self, a: np.ndarray, I: np.ndarray, c: np.ndarray):
        """Plot individual cost-to-go components."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, a[:, 0, 0], color=self.colors[0], label=r'$\hat{x}^T \cdot S \cdot \hat{x}$')
        ax.plot(self.time, I[:, 0, 0], color=self.colors[1], label=r'$\int Tr(K \cdot M \cdot K^T \cdot S)$')
        ax.plot(self.time, c[:, 0, 0], color=self.colors[2], label=r'$Tr(P_f \cdot S_f)$')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Optimal Cost-to-go')
        ax.set_title('Cost-to-Go Components')
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1.0)
        plt.tight_layout()
        plt.savefig('cost_components.png', dpi=150, bbox_inches='tight')
        plt.show()


# =============================================================================
# Monte Carlo Analysis
# =============================================================================

def run_monte_carlo(params: SystemParameters, K_control: np.ndarray, 
                    K_kalman: np.ndarray, time_array: np.ndarray,
                    n_runs: int = 500) -> dict:
    """
    Run Monte Carlo simulation to evaluate statistical performance.
    
    Args:
        params: System parameters
        K_control: Pre-computed control gains
        K_kalman: Pre-computed Kalman gains
        time_array: Time array
        n_runs: Number of Monte Carlo runs
        
    Returns:
        Dictionary with statistical results
    """
    miss_distances = []
    final_costs = []
    
    print(f"\nRunning Monte Carlo simulation with {n_runs} runs...")
    
    for i in range(n_runs):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs")
        
        simulator = GuidanceSimulator(params, time_array)
        X, X_hat, U = simulator.run(K_control, K_kalman)
        
        miss_distance = X[-1, 0, 0]
        miss_distances.append(miss_distance)
        
        cost_calc = CostCalculator(params, time_array)
        final_cost = cost_calc.compute_final_cost(X, U)
        final_costs.append(final_cost)
    
    miss_distances = np.array(miss_distances)
    final_costs = np.array(final_costs)
    
    results = {
        'miss_distances': miss_distances,
        'final_costs': final_costs,
        'mean_miss': np.mean(miss_distances),
        'std_miss': np.std(miss_distances),
        'rms_miss': np.sqrt(np.mean(miss_distances**2)),
        'mean_cost': np.mean(final_costs),
        'std_cost': np.std(final_costs),
    }
    
    print(f"\nMonte Carlo Results ({n_runs} runs):")
    print(f"  Mean miss distance: {results['mean_miss']:.4f} m")
    print(f"  Std miss distance:  {results['std_miss']:.4f} m")
    print(f"  RMS miss distance:  {results['rms_miss']:.4f} m")
    print(f"  Mean final cost:    {results['mean_cost']:.4f}")
    print(f"  Std final cost:     {results['std_cost']:.4f}")
    
    return results


def plot_monte_carlo_results(results: dict):
    """Plot Monte Carlo simulation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Miss distance histogram
    axes[0].hist(results['miss_distances'], bins=50, color=COLORS[0], 
                 edgecolor='white', alpha=0.8)
    axes[0].axvline(results['mean_miss'], color=COLORS[1], linestyle='--', 
                    linewidth=2, label=f'Mean: {results["mean_miss"]:.2f} m')
    axes[0].axvline(results['rms_miss'], color=COLORS[2], linestyle=':', 
                    linewidth=2, label=f'RMS: {results["rms_miss"]:.2f} m')
    axes[0].set_xlabel('Miss Distance [m]')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Miss Distance Distribution')
    axes[0].legend()
    
    # Final cost histogram
    axes[1].hist(results['final_costs'], bins=50, color=COLORS[3], 
                 edgecolor='white', alpha=0.8)
    axes[1].axvline(results['mean_cost'], color=COLORS[1], linestyle='--', 
                    linewidth=2, label=f'Mean: {results["mean_cost"]:.2f}')
    axes[1].set_xlabel('Final Cost')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Final Cost Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def run_optimal_missile_guidance():
    """Main function to run the complete guidance simulation."""
    
    # Initialize system parameters
    params = SystemParameters()
    
    print("="*60)
    print("Optimal Missile Guidance - LQG Control")
    print("="*60)
    print(f"\nParameters:")
    print(f"  b (control weight factor): {params.b}")
    print(f"  R (= b/2):                 {params.R}")
    print(f"  V (closing velocity):      {params.V} m/s")
    print(f"  t_f (final time):          {params.t_final} s")
    
    # Solve Riccati equations
    riccati_solver = RiccatiSolver(params)
    K_control, K_kalman, S, P = riccati_solver.solve()
    time = riccati_solver.time_array
    
    # Run single simulation
    print("\n" + "="*60)
    print("Single Run Simulation")
    print("="*60)
    
    simulator = GuidanceSimulator(params, time)
    X, X_hat, U = simulator.run(K_control, K_kalman)
    
    # Calculate costs
    cost_calc = CostCalculator(params, time)
    a, b, c, I, J = cost_calc.compute_cost_to_go(X_hat, S, K_kalman, P)
    final_cost = cost_calc.compute_final_cost(X, U)
    
    # Print results
    print(f"\nSingle Run Results:")
    print(f"  Final position (miss distance): {X[-1, 0, 0]:.4f} m")
    print(f"  Final cost: {final_cost:.4f}")
    
    # Create plots
    plotter = ResultsPlotter(time)
    plotter.plot_control_gains(K_control)
    plotter.plot_covariance(P)
    plotter.plot_kalman_gains(K_kalman)
    plotter.plot_states(X, X_hat)
    plotter.plot_control_input(U)
    plotter.plot_cost_to_go(J)
    plotter.plot_cost_components(a, I, c)
    
    # Run Monte Carlo simulation
    print("\n" + "="*60)
    print("Monte Carlo Analysis")
    print("="*60)
    
    mc_results = run_monte_carlo(params, K_control, K_kalman, time, n_runs=500)
    plot_monte_carlo_results(mc_results)
    
    return {
        'params': params,
        'K_control': K_control,
        'K_kalman': K_kalman,
        'S': S,
        'P': P,
        'time': time,
        'X': X,
        'X_hat': X_hat,
        'U': U,
        'cost_components': (a, b, c, I, J),
        'monte_carlo': mc_results
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        results = run_optimal_missile_guidance()
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()