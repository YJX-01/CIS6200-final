"""
Utility functions for coalitional congestion game experiments.

Includes:
- Resource generation
- Coalition structure utilities
- Metrics calculation
- Data management
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import json
from pathlib import Path
from game import Resource, CongestionGame, CoalitionStructure


def create_standard_resources(
    resource_type: str = 'heterogeneous'
) -> List[Resource]:
    """
    Create standard resource configurations for experiments.
    
    Args:
        resource_type: Type of resources to create
            - 'heterogeneous': Mix of tight/forgiving/medium (from paper)
            - 'homogeneous': All identical resources
            - 'random': Random parameters
            
    Returns:
        List of Resource objects
    """
    if resource_type == 'heterogeneous':
        # Configuration from paper Section 4.2
        return [
            Resource(id=0, a=1.0, b=0.0),   # Tight: high congestion sensitivity
            Resource(id=1, a=0.5, b=2.0),   # Forgiving: low sensitivity, high base
            Resource(id=2, a=1.5, b=1.0),   # Medium: intermediate
        ]
    
    elif resource_type == 'homogeneous':
        return [
            Resource(id=i, a=1.0, b=1.0)
            for i in range(3)
        ]
    
    elif resource_type == 'random':
        return [
            Resource(
                id=i,
                a=np.random.uniform(1.0, 10.0),
                b=np.random.uniform(0.0, 5.0)
            )
            for i in range(3)
        ]
    
    else:
        raise ValueError(f"Unknown resource type: {resource_type}")


def create_parameter_grid(
    rho_values: Optional[List[float]] = None,
    sigma_values: Optional[List[float]] = None
) -> List[Tuple[float, float]]:
    """
    Create parameter grid for (rho, sigma) experiments.
    
    Args:
        rho_values: List of rho values (altruism)
        sigma_values: List of sigma values (spite)
        
    Returns:
        List of (rho, sigma) tuples
    """
    if rho_values is None:
        rho_values = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    
    if sigma_values is None:
        sigma_values = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    
    return [(rho, sigma) for rho in rho_values for sigma in sigma_values]


def calculate_potential_bound(
    resources: List[Resource],
    n_players: int
) -> float:
    """
    Calculate the potential-compatibility bound from Proposition 1.
    
    Bound: rho + sigma <= alpha_min / (4 * n^2 * alpha_max + 4 * n * (beta_max - beta_min))
    
    Args:
        resources: List of resources
        n_players: Number of players
        
    Returns:
        Upper bound for rho + sigma
    """
    alpha_min = min(r.a for r in resources)
    alpha_max = max(r.a for r in resources)
    
    beta_min = min(r.b for r in resources)
    beta_max = max(r.b for r in resources)
    
    return alpha_min / (4 * n_players**2 * alpha_max + 4 * n_players * (beta_max - beta_min))


def classify_regime(
    rho: float,
    sigma: float,
    bound: float,
    threshold_high: float = 0.5
) -> str:
    """
    Classify (rho, sigma) into one of four regimes.
    
    Regimes from Section 4.1:
    1. Nearly selfish: rho + sigma <= bound
    2. Altruistic: rho large, sigma small
    3. Spiteful: sigma large, rho small
    4. Factional: both large and comparable
    
    Args:
        rho: In-group altruism
        sigma: Out-group spite
        bound: Potential-compatibility bound
        threshold_high: Threshold for "large" values
        
    Returns:
        Regime name
    """
    if rho + sigma <= bound:
        return 'nearly_selfish'
    elif rho >= threshold_high and sigma < 0.3 * rho:
        return 'altruistic'
    elif sigma >= threshold_high and rho < 0.3 * sigma:
        return 'spiteful'
    elif rho >= threshold_high and sigma >= threshold_high:
        return 'factional'
    else:
        return 'mixed'


def coalition_structure_metrics(structure: CoalitionStructure) -> Dict:
    """
    Calculate descriptive metrics for a coalition structure.
    
    Args:
        structure: Coalition structure
        
    Returns:
        Dictionary of metrics
    """
    sizes = [len(c) for c in structure]
    
    return {
        'n_coalitions': len(structure),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'mean_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'is_singleton': len(structure) == sum(sizes),
        'is_grand': len(structure) == 1,
    }


def strategy_profile_metrics(
    game: CongestionGame,
    strategy_profile: np.ndarray
) -> Dict:
    """
    Calculate metrics for a strategy profile.
    
    Args:
        game: CongestionGame instance
        strategy_profile: Strategy profile
        
    Returns:
        Dictionary of metrics
    """
    congestion = game.get_congestion(strategy_profile)
    
    return {
        'social_cost': game.social_cost(strategy_profile),
        'rosenthal_potential': game.rosenthal_potential(strategy_profile),
        'max_congestion': int(np.max(congestion)),
        'min_congestion': int(np.min(congestion)),
        'congestion_std': float(np.std(congestion)),
        'congestion_distribution': congestion.tolist(),
    }


def calculate_price_of_anarchy(
    nash_cost: float,
    optimal_cost: float
) -> float:
    """
    Calculate Price of Anarchy (PoA).
    
    PoA = cost of worst Nash / cost of optimal
    
    Args:
        nash_cost: Social cost at Nash equilibrium
        optimal_cost: Social cost at optimum
        
    Returns:
        Price of Anarchy
    """
    if optimal_cost == 0:
        return float('inf')
    return nash_cost / optimal_cost


def coalition_structure_to_string(structure: CoalitionStructure) -> str:
    """
    Convert coalition structure to readable string.
    
    Args:
        structure: Coalition structure
        
    Returns:
        String representation
    """
    sorted_structure = [sorted(list(c)) for c in structure]
    sorted_structure.sort()
    return str(sorted_structure)


def coalition_structure_from_string(s: str) -> CoalitionStructure:
    """
    Parse coalition structure from string.
    
    Args:
        s: String representation
        
    Returns:
        Coalition structure
    """
    parsed = eval(s)  # Safe here since we control the format
    return [set(c) for c in parsed]


class ExperimentLogger:
    """Logger for experiment results."""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def log_result(self, result: Dict):
        """
        Log a single experiment result.
        
        Args:
            result: Dictionary of results
        """
        # Convert sets to lists for JSON serialization
        result_serializable = self._make_serializable(result)
        self.results.append(result_serializable)
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def save_results(self, filename: str = 'results.json'):
        """
        Save all results to file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def load_results(self, filename: str = 'results.json'):
        """
        Load results from file.
        
        Args:
            filename: Input filename
        """
        input_path = self.output_dir / filename
        with open(input_path, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} results from {input_path}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics across all results.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {}
        
        # Extract numeric fields
        convergence_rates = []
        nash_eq_counts = []
        split_stable_rates = []
        join_stable_rates = []
        coalition_eq_rates = []
        
        for result in self.results:
            if 'converged' in result:
                convergence_rates.append(float(result['converged']))
            if 'n_nash_equilibria' in result:
                nash_eq_counts.append(result['n_nash_equilibria'])
            if 'split_stable' in result:
                split_stable_rates.append(float(result['split_stable']))
            if 'join_stable' in result:
                join_stable_rates.append(float(result['join_stable']))
            if 'is_coalition_equilibrium' in result:
                coalition_eq_rates.append(float(result['is_coalition_equilibrium']))
        
        summary = {}
        
        if convergence_rates:
            summary['convergence_rate'] = np.mean(convergence_rates)
        if nash_eq_counts:
            summary['mean_nash_count'] = np.mean(nash_eq_counts)
            summary['median_nash_count'] = np.median(nash_eq_counts)
        if split_stable_rates:
            summary['split_stable_rate'] = np.mean(split_stable_rates)
        if join_stable_rates:
            summary['join_stable_rate'] = np.mean(join_stable_rates)
        if coalition_eq_rates:
            summary['coalition_eq_rate'] = np.mean(coalition_eq_rates)
        
        return summary


def pretty_print_game_state(
    game: CongestionGame,
    strategy_profile: np.ndarray,
    title: str = "Game State"
):
    """
    Pretty print current game state.
    
    Args:
        game: CongestionGame instance
        strategy_profile: Current strategy profile
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\nParameters: rho={game.rho:.2f}, sigma={game.sigma:.2f}")
    
    print(f"\nCoalition Structure:")
    for i, coalition in enumerate(game.coalition_structure):
        print(f"  Coalition {i}: {sorted(list(coalition))}")
    
    print(f"\nStrategy Profile:")
    congestion = game.get_congestion(strategy_profile)
    for k, resource in enumerate(game.resources):
        players_on_k = [i for i in range(game.n_players) if strategy_profile[i] == k]
        latency = resource.latency(congestion[k])
        print(f"  Resource {k} (a={resource.a}, b={resource.b}): "
              f"{len(players_on_k)} players {players_on_k}, latency={latency:.2f}")
    
    print(f"\nPlayer Utilities:")
    for i in range(game.n_players):
        utility = game.get_utility(i, strategy_profile)
        latency = game.get_latency(i, strategy_profile)
        print(f"  Player {i}: utility={utility:.2f}, latency={latency:.2f}")
    
    print(f"\nCoalition Utilities:")
    for i, coalition in enumerate(game.coalition_structure):
        utility = game.get_coalition_utility(coalition, strategy_profile)
        print(f"  Coalition {i}: utility={utility:.2f}")
    
    print(f"\nAggregate Metrics:")
    print(f"  Social cost: {game.social_cost(strategy_profile):.2f}")
    print(f"  Rosenthal potential: {game.rosenthal_potential(strategy_profile):.2f}")
    
    print(f"{'='*60}\n")

