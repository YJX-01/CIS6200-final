"""
Nash equilibrium computation for coalitional congestion games.

Implements:
- Best-response dynamics
- Exhaustive Nash equilibrium search
- Logit learning dynamics
- Potential-compatibility checking
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
import itertools
from game import CongestionGame, CoalitionStructure


class EquilibriumSolver:
    """Solver for Nash equilibria in coalitional congestion games."""
    
    def __init__(self, game: CongestionGame, max_iterations: int = 1000):
        """
        Initialize equilibrium solver.
        
        Args:
            game: CongestionGame instance
            max_iterations: Maximum iterations for best-response dynamics
        """
        self.game = game
        self.max_iterations = max_iterations
    
    def best_response(
        self, 
        coalition: Set[int], 
        strategy_profile: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Find best joint response for a coalition given others' strategies.
        
        Args:
            coalition: Set of players in the coalition
            strategy_profile: Current strategy profile
            
        Returns:
            (best_joint_action, best_utility)
        """
        coalition_list = sorted(list(coalition))
        best_utility = self.game.get_coalition_utility(coalition, strategy_profile)
        best_action = strategy_profile[coalition_list].copy()
        
        # Enumerate all possible joint actions for the coalition
        for joint_action in itertools.product(
            range(self.game.n_resources), 
            repeat=len(coalition)
        ):
            # Create new strategy profile with this joint action
            new_profile = strategy_profile.copy()
            for idx, player in enumerate(coalition_list):
                new_profile[player] = joint_action[idx]
            
            # Calculate utility
            utility = self.game.get_coalition_utility(coalition, new_profile)
            
            if utility > best_utility:
                best_utility = utility
                best_action = np.array(joint_action)
        
        return best_action, best_utility
    
    def is_nash_equilibrium(self, strategy_profile: np.ndarray) -> bool:
        """
        Check if a strategy profile is a coalitional Nash equilibrium.
        
        A profile is a Nash equilibrium if no coalition can profitably deviate.
        
        Args:
            strategy_profile: Strategy profile to check
            
        Returns:
            True if Nash equilibrium, False otherwise
        """
        for coalition in self.game.coalition_structure:
            current_utility = self.game.get_coalition_utility(
                coalition, strategy_profile
            )
            
            best_action, best_utility = self.best_response(
                coalition, strategy_profile
            )
            
            # Allow small numerical tolerance
            if best_utility > current_utility + 1e-9:
                return False
        
        return True
    
    def best_response_dynamics(
        self, 
        initial_profile: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, bool, int, List[float]]:
        """
        Run best-response dynamics until convergence or max iterations.
        
        Args:
            initial_profile: Starting strategy profile (random if None)
            verbose: Whether to print iteration details
            
        Returns:
            (final_profile, converged, iterations, potential_trajectory)
        """
        if initial_profile is None:
            strategy_profile = np.random.randint(
                0, self.game.n_resources, size=self.game.n_players
            )
        else:
            strategy_profile = initial_profile.copy()
        
        potential_trajectory = [self.game.rosenthal_potential(strategy_profile)]
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Iterate through coalitions
            for coalition in self.game.coalition_structure:
                coalition_list = sorted(list(coalition))
                current_utility = self.game.get_coalition_utility(
                    coalition, strategy_profile
                )
                
                best_action, best_utility = self.best_response(
                    coalition, strategy_profile
                )
                
                # If improvement found, update strategy
                if best_utility > current_utility + 1e-9:
                    for idx, player in enumerate(coalition_list):
                        strategy_profile[player] = best_action[idx]
                    improved = True
                    
                    if verbose:
                        print(f"Iteration {iteration}: Coalition {coalition} "
                              f"improved from {current_utility:.4f} to {best_utility:.4f}")
            
            potential_trajectory.append(
                self.game.rosenthal_potential(strategy_profile)
            )
            
            # Check convergence
            if not improved:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                return strategy_profile, True, iteration + 1, potential_trajectory
        
        if verbose:
            print(f"Did not converge after {self.max_iterations} iterations")
        return strategy_profile, False, self.max_iterations, potential_trajectory
    
    def find_all_nash_equilibria(
        self, 
        max_profiles: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Find all pure Nash equilibria by exhaustive search.
        
        Warning: Exponential complexity! Only feasible for small games.
        
        Args:
            max_profiles: Maximum number of profiles to check (None = check all)
            
        Returns:
            List of Nash equilibrium strategy profiles
        """
        equilibria = []
        n_profiles = self.game.n_resources ** self.game.n_players
        
        if max_profiles is not None:
            n_profiles = min(n_profiles, max_profiles)
        
        # Enumerate all possible strategy profiles
        for profile_tuple in itertools.product(
            range(self.game.n_resources), 
            repeat=self.game.n_players
        ):
            profile = np.array(profile_tuple)
            
            if self.is_nash_equilibrium(profile):
                equilibria.append(profile)
            
            if len(equilibria) >= n_profiles:
                break
        
        return equilibria
    
    def check_potential_compatibility(
        self, 
        strategy_profile: np.ndarray,
        deviation_coalition: Set[int],
        new_coalition_actions: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Check if a deviation satisfies potential-compatibility condition.
        
        Potential-compatible means:
            ΔU_C ≥ 0  =>  ΔΦ^0 ≤ 0
        
        Args:
            strategy_profile: Current strategy profile
            deviation_coalition: Coalition that deviates
            new_coalition_actions: New actions for deviating coalition
            
        Returns:
            (is_compatible, delta_utility, delta_potential)
        """
        # Current state
        current_utility = self.game.get_coalition_utility(
            deviation_coalition, strategy_profile
        )
        current_potential = self.game.rosenthal_potential(strategy_profile)
        
        # New state after deviation
        new_profile = strategy_profile.copy()
        coalition_list = sorted(list(deviation_coalition))
        for idx, player in enumerate(coalition_list):
            new_profile[player] = new_coalition_actions[idx]
        
        new_utility = self.game.get_coalition_utility(
            deviation_coalition, new_profile
        )
        new_potential = self.game.rosenthal_potential(new_profile)
        
        delta_utility = new_utility - current_utility
        delta_potential = new_potential - current_potential
        
        # Check compatibility: if utility improves, potential should decrease
        is_compatible = not (delta_utility > 1e-9 and delta_potential > 1e-9)
        
        return is_compatible, delta_utility, delta_potential


class LogitLearning:
    """Logit learning dynamics for coalitional congestion games."""
    
    def __init__(
        self, 
        game: CongestionGame, 
        temperature: float = 0.1
    ):
        """
        Initialize logit learning.
        
        Args:
            game: CongestionGame instance
            temperature: Temperature parameter τ (lower = more rational)
        """
        self.game = game
        self.temperature = temperature
    
    def coalition_logit_response(
        self, 
        coalition: Set[int], 
        strategy_profile: np.ndarray
    ) -> np.ndarray:
        """
        Sample a joint action for coalition according to logit probabilities.
        
        Probability of joint action a is proportional to exp(U_C(a, s_{-C}) / τ).
        
        Args:
            coalition: Coalition that updates
            strategy_profile: Current strategy profile
            
        Returns:
            Sampled joint action for the coalition
        """
        coalition_list = sorted(list(coalition))
        n_coalition = len(coalition)
        
        # Enumerate all joint actions and their utilities
        utilities = []
        actions = []
        
        for joint_action in itertools.product(
            range(self.game.n_resources), 
            repeat=n_coalition
        ):
            # Create profile with this joint action
            test_profile = strategy_profile.copy()
            for idx, player in enumerate(coalition_list):
                test_profile[player] = joint_action[idx]
            
            utility = self.game.get_coalition_utility(coalition, test_profile)
            utilities.append(utility)
            actions.append(joint_action)
        
        # Convert to probabilities via softmax
        utilities = np.array(utilities)
        exp_utilities = np.exp(utilities / self.temperature)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # Sample action
        chosen_idx = np.random.choice(len(actions), p=probabilities)
        return np.array(actions[chosen_idx])
    
    def run_dynamics(
        self, 
        n_iterations: int,
        initial_profile: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run logit learning dynamics for specified iterations.
        
        Args:
            n_iterations: Number of iterations to run
            initial_profile: Starting profile (random if None)
            
        Returns:
            (trajectory of profiles, trajectory of potentials)
        """
        if initial_profile is None:
            current_profile = np.random.randint(
                0, self.game.n_resources, size=self.game.n_players
            )
        else:
            current_profile = initial_profile.copy()
        
        trajectory = [current_profile.copy()]
        potentials = [self.game.rosenthal_potential(current_profile)]
        
        for t in range(n_iterations):
            # Randomly select a coalition to update
            coalition = self.game.coalition_structure[
                np.random.randint(len(self.game.coalition_structure))
            ]
            
            # Sample new action via logit response
            new_action = self.coalition_logit_response(coalition, current_profile)
            
            # Update profile
            coalition_list = sorted(list(coalition))
            for idx, player in enumerate(coalition_list):
                current_profile[player] = new_action[idx]
            
            trajectory.append(current_profile.copy())
            potentials.append(self.game.rosenthal_potential(current_profile))
        
        return trajectory, potentials
    
    def stationary_distribution(
        self, 
        n_samples: int = 10000,
        burn_in: int = 1000
    ) -> Dict[Tuple[int, ...], float]:
        """
        Estimate stationary distribution via sampling.
        
        Args:
            n_samples: Number of samples to collect
            burn_in: Initial iterations to discard
            
        Returns:
            Dictionary mapping strategy profiles to empirical frequencies
        """
        # Run burn-in
        trajectory, _ = self.run_dynamics(burn_in)
        current_profile = trajectory[-1]
        
        # Collect samples
        trajectory, _ = self.run_dynamics(n_samples, initial_profile=current_profile)
        
        # Count frequencies
        frequency_dict = {}
        for profile in trajectory:
            profile_tuple = tuple(profile)
            frequency_dict[profile_tuple] = frequency_dict.get(profile_tuple, 0) + 1
        
        # Normalize to probabilities
        total = sum(frequency_dict.values())
        return {k: v / total for k, v in frequency_dict.items()}

