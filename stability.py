"""
Coalition stability testing for congestion games.

Implements:
- Split deviation testing
- Join deviation testing
- Coalition equilibrium checking
- Structural stability analysis
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from itertools import combinations
from game import CongestionGame, CoalitionStructure


class StabilityChecker:
    """Check stability of coalition structures in congestion games."""
    
    def __init__(self, game: CongestionGame):
        """
        Initialize stability checker.
        
        Args:
            game: CongestionGame instance
        """
        self.game = game
    
    def generate_split(
        self, 
        coalition: Set[int]
    ) -> List[Tuple[Set[int], Set[int]]]:
        """
        Generate all possible binary splits of a coalition.
        
        Args:
            coalition: Coalition to split
            
        Returns:
            List of (subset_a, subset_b) pairs
        """
        if len(coalition) <= 1:
            return []
        
        splits = []
        coalition_list = sorted(list(coalition))
        
        # Generate all non-empty proper subsets
        for size in range(1, len(coalition_list)):
            for subset_a_tuple in combinations(coalition_list, size):
                subset_a = set(subset_a_tuple)
                subset_b = coalition - subset_a
                
                # Avoid duplicates (only include one ordering)
                if min(subset_a) < min(subset_b):
                    splits.append((subset_a, subset_b))
        
        return splits
    
    def apply_split(
        self, 
        coalition_structure: CoalitionStructure,
        original_coalition: Set[int],
        split_a: Set[int],
        split_b: Set[int]
    ) -> CoalitionStructure:
        """
        Create new coalition structure by splitting a coalition.
        
        Args:
            coalition_structure: Original structure
            original_coalition: Coalition to split
            split_a: First subset
            split_b: Second subset
            
        Returns:
            New coalition structure with split applied
        """
        new_structure = []
        for coalition in coalition_structure:
            if coalition == original_coalition:
                new_structure.append(split_a)
                new_structure.append(split_b)
            else:
                new_structure.append(coalition)
        return new_structure
    
    def apply_join(
        self,
        coalition_structure: CoalitionStructure,
        coalition_a: Set[int],
        coalition_b: Set[int]
    ) -> CoalitionStructure:
        """
        Create new coalition structure by joining two coalitions.
        
        Args:
            coalition_structure: Original structure
            coalition_a: First coalition to merge
            coalition_b: Second coalition to merge
            
        Returns:
            New coalition structure with join applied
        """
        new_structure = []
        merged = coalition_a | coalition_b
        merged_added = False
        
        for coalition in coalition_structure:
            if coalition == coalition_a or coalition == coalition_b:
                if not merged_added:
                    new_structure.append(merged)
                    merged_added = True
            else:
                new_structure.append(coalition)
        
        return new_structure
    
    def test_split_stability(
        self, 
        strategy_profile: np.ndarray,
        verbose: bool = False
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Test if coalition structure is stable against split deviations.
        
        A coalition structure is split-stable if no coalition can profitably split.
        
        Args:
            strategy_profile: Current strategy profile (assumed to be Nash)
            verbose: Whether to print details
            
        Returns:
            (is_stable, deviation_info)
            deviation_info contains details if profitable split found
        """
        for coalition in self.game.coalition_structure:
            if len(coalition) <= 1:
                continue
            
            # Current utility
            current_utility = self.game.get_coalition_utility(
                coalition, strategy_profile
            )
            
            # Try all possible splits
            for split_a, split_b in self.generate_split(coalition):
                # Create new game with split applied
                new_structure = self.apply_split(
                    self.game.coalition_structure,
                    coalition,
                    split_a,
                    split_b
                )
                new_game = self.game.copy_with_new_coalition_structure(new_structure)
                
                # Calculate utilities under new structure
                utility_a = new_game.get_coalition_utility(split_a, strategy_profile)
                utility_b = new_game.get_coalition_utility(split_b, strategy_profile)
                total_utility = utility_a + utility_b
                
                # Check if split is profitable
                if total_utility > current_utility + 1e-9:
                    if verbose:
                        print(f"Profitable split found:")
                        print(f"  Original coalition: {coalition} (utility: {current_utility:.4f})")
                        print(f"  Split into: {split_a} (utility: {utility_a:.4f})")
                        print(f"            + {split_b} (utility: {utility_b:.4f})")
                        print(f"  Total gain: {total_utility - current_utility:.4f}")
                    
                    return False, {
                        'type': 'split',
                        'original_coalition': coalition,
                        'split_a': split_a,
                        'split_b': split_b,
                        'original_utility': current_utility,
                        'new_utility_a': utility_a,
                        'new_utility_b': utility_b,
                        'gain': total_utility - current_utility
                    }
        
        return True, None
    
    def test_join_stability(
        self, 
        strategy_profile: np.ndarray,
        verbose: bool = False
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Test if coalition structure is stable against join deviations.
        
        A coalition structure is join-stable if no pair of coalitions
        can profitably merge.
        
        Args:
            strategy_profile: Current strategy profile (assumed to be Nash)
            verbose: Whether to print details
            
        Returns:
            (is_stable, deviation_info)
            deviation_info contains details if profitable join found
        """
        n_coalitions = len(self.game.coalition_structure)
        
        # Try all pairs of coalitions
        for i in range(n_coalitions):
            for j in range(i + 1, n_coalitions):
                coalition_a = self.game.coalition_structure[i]
                coalition_b = self.game.coalition_structure[j]
                
                # Current utilities
                utility_a = self.game.get_coalition_utility(
                    coalition_a, strategy_profile
                )
                utility_b = self.game.get_coalition_utility(
                    coalition_b, strategy_profile
                )
                current_total = utility_a + utility_b
                
                # Create new game with join applied
                new_structure = self.apply_join(
                    self.game.coalition_structure,
                    coalition_a,
                    coalition_b
                )
                new_game = self.game.copy_with_new_coalition_structure(new_structure)
                
                # Calculate utility of merged coalition
                merged_coalition = coalition_a | coalition_b
                merged_utility = new_game.get_coalition_utility(
                    merged_coalition, strategy_profile
                )
                
                # Check if join is profitable
                if merged_utility > current_total + 1e-9:
                    if verbose:
                        print(f"Profitable join found:")
                        print(f"  Coalition A: {coalition_a} (utility: {utility_a:.4f})")
                        print(f"  Coalition B: {coalition_b} (utility: {utility_b:.4f})")
                        print(f"  Merged: {merged_coalition} (utility: {merged_utility:.4f})")
                        print(f"  Total gain: {merged_utility - current_total:.4f}")
                    
                    return False, {
                        'type': 'join',
                        'coalition_a': coalition_a,
                        'coalition_b': coalition_b,
                        'merged_coalition': merged_coalition,
                        'utility_a': utility_a,
                        'utility_b': utility_b,
                        'merged_utility': merged_utility,
                        'gain': merged_utility - current_total
                    }
        
        return True, None
    
    def is_coalition_equilibrium(
        self, 
        strategy_profile: np.ndarray,
        check_nash: bool = True,
        verbose: bool = False
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Check if (P, s) is a coalition equilibrium.
        
        Definition: A pair (P, s) is a coalition equilibrium if:
        (i) Action stability: s is a coalitional Nash equilibrium given P
        (ii) Structural stability: P is stable against splits and joins
        
        Args:
            strategy_profile: Strategy profile to check
            check_nash: Whether to verify Nash equilibrium property
            verbose: Whether to print details
            
        Returns:
            (is_equilibrium, failure_reason, deviation_info)
        """
        # Check action stability (Nash equilibrium)
        if check_nash:
            from equilibrium import EquilibriumSolver
            solver = EquilibriumSolver(self.game)
            
            if not solver.is_nash_equilibrium(strategy_profile):
                if verbose:
                    print("Failed action stability: Not a Nash equilibrium")
                return False, "action_unstable", None
        
        # Check split stability
        split_stable, split_info = self.test_split_stability(
            strategy_profile, verbose=verbose
        )
        if not split_stable:
            return False, "split_unstable", split_info
        
        # Check join stability
        join_stable, join_info = self.test_join_stability(
            strategy_profile, verbose=verbose
        )
        if not join_stable:
            return False, "join_unstable", join_info
        
        if verbose:
            print("Coalition equilibrium verified!")
        return True, "coalition_equilibrium", None


def enumerate_coalition_structures(n_players: int) -> List[CoalitionStructure]:
    """
    Enumerate all possible coalition structures (set partitions).
    
    Warning: Number of partitions is the Bell number B(n), which grows rapidly!
    B(6) = 203, B(7) = 877, B(8) = 4140, etc.
    
    Args:
        n_players: Number of players
        
    Returns:
        List of all coalition structures
    """
    def partitions_recursive(items: List[int]) -> List[List[Set[int]]]:
        """Recursive partition generator."""
        if len(items) == 0:
            return [[]]
        
        first = items[0]
        rest = items[1:]
        
        all_partitions = []
        
        # Generate partitions of rest
        for smaller_partition in partitions_recursive(rest):
            # Add first element to each existing subset
            for i, subset in enumerate(smaller_partition):
                new_partition = [s.copy() for s in smaller_partition]
                new_partition[i].add(first)
                all_partitions.append(new_partition)
            
            # Create new subset with just first element
            new_partition = smaller_partition + [{first}]
            all_partitions.append(new_partition)
        
        return all_partitions
    
    return partitions_recursive(list(range(n_players)))


def sample_random_coalition_structures(
    n_players: int, 
    n_samples: int,
    seed: Optional[int] = None
) -> List[CoalitionStructure]:
    """
    Sample random coalition structures uniformly.
    
    Uses a simple algorithm: assign each player randomly to a coalition.
    
    Args:
        n_players: Number of players
        n_samples: Number of structures to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled coalition structures
    """
    if seed is not None:
        np.random.seed(seed)
    
    structures = []
    
    for _ in range(n_samples):
        # Random partition: assign each player to a random coalition ID
        coalition_ids = np.random.randint(0, n_players, size=n_players)
        
        # Convert to coalition structure
        structure_dict = {}
        for player, coalition_id in enumerate(coalition_ids):
            if coalition_id not in structure_dict:
                structure_dict[coalition_id] = set()
            structure_dict[coalition_id].add(player)
        
        structure = list(structure_dict.values())
        structures.append(structure)
    
    return structures


def create_standard_structures(n_players: int) -> Dict[str, CoalitionStructure]:
    """
    Create standard coalition structures for testing.
    
    Args:
        n_players: Number of players
        
    Returns:
        Dictionary of named coalition structures
    """
    structures = {}
    
    # Singleton: each player alone
    structures['singleton'] = [{i} for i in range(n_players)]
    
    # Grand coalition: all players together
    structures['grand'] = [set(range(n_players))]
    
    # Pairs (if even number of players)
    if n_players % 2 == 0:
        structures['pairs'] = [
            {2*i, 2*i + 1} for i in range(n_players // 2)
        ]
    
    # Half-split (if even)
    if n_players % 2 == 0:
        structures['half_split'] = [
            set(range(n_players // 2)),
            set(range(n_players // 2, n_players))
        ]
    
    # Thirds (if divisible by 3)
    if n_players % 3 == 0:
        third = n_players // 3
        structures['thirds'] = [
            set(range(0, third)),
            set(range(third, 2*third)),
            set(range(2*third, n_players))
        ]
    
    # Asymmetric: one large + several small
    if n_players >= 4:
        structures['asymmetric'] = [{0}, set(range(1, n_players))]
    
    return structures

