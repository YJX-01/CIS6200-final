import numpy as np
from typing import List, Tuple, Dict, Set, Callable, Optional, FrozenSet
from dataclasses import dataclass
import itertools


@dataclass
class Resource:
    """Represents a resource with a linear latency function f_k(n) = a*n + b."""
    id: int
    a: float  # slope (congestion sensitivity)
    b: float  # base latency
    
    def latency(self, congestion: int) -> float:
        """Calculate latency for given congestion level."""
        return self.a * congestion + self.b


# Type alias for coalition structure: List of sets of player indices
CoalitionStructure = List[Set[int]]


class CongestionGame:
    """
    Congestion game with social preferences.
    
    Model from the paper:
    - N players choose from K resources (singleton strategies)
    - Linear latency functions f_k(n) = a_k*n + b_k
    - Coalition structure P partitions players
    - Social preferences: rho (in-group altruism), sigma (out-group spite)
    
    Utility formula (Eq. 1):
        U_i(s;P) = μ_i(s) + rho·Σ_{j∈C_i\{i}} μ_j(s) - sigma·Σ_{m∉C_i} μ_m(s)
    where μ_i(s) = -ℓ_i(s) is selfish payoff.
    """
    
    def __init__(
        self, 
        n_players: int, 
        resources: List[Resource],
        coalition_structure: CoalitionStructure,
        rho: float = 0.0,
        sigma: float = 0.0
    ):
        """
        Initialize congestion game.
        
        Args:
            n_players: Number of players
            resources: List of Resource objects
            coalition_structure: Partition of players into coalitions
            rho: In-group altruism parameter (≥0)
            sigma: Out-group spite parameter (≥0)
        """
        self.n_players = n_players
        self.players = list(range(n_players))
        self.resources = resources
        self.n_resources = len(resources)
        self.coalition_structure = coalition_structure
        self.rho = rho
        self.sigma = sigma
        
        # Build player -> coalition mapping for efficient lookup
        self._player_to_coalition = {}
        for coalition_idx, coalition in enumerate(coalition_structure):
            for player in coalition:
                self._player_to_coalition[player] = coalition_idx
        
        # Validate coalition structure
        self._validate_coalition_structure()
    
    def _validate_coalition_structure(self):
        """Ensure coalition structure is a valid partition."""
        all_players = set()
        for coalition in self.coalition_structure:
            all_players.update(coalition)
        
        assert all_players == set(self.players), \
            f"Coalition structure must partition all players. Missing: {set(self.players) - all_players}"
        assert len(all_players) == self.n_players, \
            "Coalition structure contains duplicate players"
    
    def get_player_coalition(self, player: int) -> Set[int]:
        """Get the coalition containing a given player."""
        coalition_idx = self._player_to_coalition[player]
        return self.coalition_structure[coalition_idx]
    
    def get_congestion(self, strategy_profile: np.ndarray) -> np.ndarray:
        """
        Calculate congestion level n_k(s) for each resource.
        
        Args:
            strategy_profile: Array of shape (n_players,) with resource choices
            
        Returns:
            Array of congestion levels for each resource
        """
        return np.bincount(strategy_profile, minlength=self.n_resources)
    
    def get_latency(self, player: int, strategy_profile: np.ndarray) -> float:
        """
        Calculate latency ℓ_i(s) for player i.
        
        Args:
            player: Player index
            strategy_profile: Current strategy profile
            
        Returns:
            Latency experienced by the player
        """
        resource_id = strategy_profile[player]
        congestion = self.get_congestion(strategy_profile)
        return self.resources[resource_id].latency(congestion[resource_id])
    
    def get_selfish_payoff(self, player: int, strategy_profile: np.ndarray) -> float:
        """
        Calculate selfish payoff μ_i(s) = -ℓ_i(s).
        
        Args:
            player: Player index
            strategy_profile: Current strategy profile
            
        Returns:
            Negative latency (higher is better)
        """
        return -self.get_latency(player, strategy_profile)
    
    def get_all_selfish_payoffs(self, strategy_profile: np.ndarray) -> np.ndarray:
        """Get selfish payoffs for all players."""
        return np.array([self.get_selfish_payoff(i, strategy_profile) 
                        for i in self.players])
    
    def get_utility(
        self, 
        player: int, 
        strategy_profile: np.ndarray
    ) -> float:
        """
        Calculate player utility U_i(s; P) with social preferences.
        
        Formula from Eq. (1):
        U_i(s;P) = μ_i(s) + rho·Σ_{j∈C_i\{i}} μ_j(s) - sigma·Σ_{m∉C_i} μ_m(s)
        
        Args:
            player: Player index
            strategy_profile: Current strategy profile
            
        Returns:
            Utility including social preferences
        """
        mu_i = self.get_selfish_payoff(player, strategy_profile)
        
        # In-group term
        player_coalition = self.get_player_coalition(player)
        in_group_sum = sum(
            self.get_selfish_payoff(j, strategy_profile)
            for j in player_coalition if j != player
        )
        
        # Out-group term
        out_group_sum = sum(
            self.get_selfish_payoff(m, strategy_profile)
            for m in self.players if m not in player_coalition
        )
        
        return mu_i + self.rho * in_group_sum - self.sigma * out_group_sum
    
    def get_coalition_utility(
        self, 
        coalition: Set[int], 
        strategy_profile: np.ndarray
    ) -> float:
        """
        Calculate coalition utility U_C(s; P) = Σ_{i∈C} U_i(s; P).
        
        Args:
            coalition: Set of player indices in the coalition
            strategy_profile: Current strategy profile
            
        Returns:
            Aggregate utility of coalition members
        """
        return sum(self.get_utility(i, strategy_profile) for i in coalition)
    
    def rosenthal_potential(self, strategy_profile: np.ndarray) -> float:
        """
        Calculate Rosenthal potential Φ^0(s) = Σ_k Σ_{x=1}^{n_k} f_k(x).
        
        This is the exact potential function for the selfish congestion game.
        
        Args:
            strategy_profile: Current strategy profile
            
        Returns:
            Potential value
        """
        congestion = self.get_congestion(strategy_profile)
        potential = 0.0
        
        for k, resource in enumerate(self.resources):
            n_k = congestion[k]
            # Sum f_k(x) for x = 1 to n_k
            for x in range(1, n_k + 1):
                potential += resource.latency(x)
        
        return potential
    
    def social_cost(self, strategy_profile: np.ndarray) -> float:
        """
        Calculate total social cost Σ_i ℓ_i(s).
        
        Args:
            strategy_profile: Current strategy profile
            
        Returns:
            Total latency across all players
        """
        return sum(self.get_latency(i, strategy_profile) for i in self.players)
    
    def copy_with_new_coalition_structure(
        self, 
        new_coalition_structure: CoalitionStructure
    ) -> 'CongestionGame':
        """
        Create a new game instance with different coalition structure.
        
        Args:
            new_coalition_structure: New partition of players
            
        Returns:
            New CongestionGame instance
        """
        return CongestionGame(
            n_players=self.n_players,
            resources=self.resources,
            coalition_structure=new_coalition_structure,
            rho=self.rho,
            sigma=self.sigma
        )

