"""
Main experiment framework for coalitional congestion games.

Implements the experimental plan:
1. Equilibrium computation across parameter space
2. Potential-compatibility analysis
3. Learning dynamics experiments
4. Stability robustness testing
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import time

from game import CongestionGame, CoalitionStructure
from equilibrium import EquilibriumSolver, LogitLearning
from stability import StabilityChecker, create_standard_structures, sample_random_coalition_structures
from utils import (
    create_standard_resources, create_parameter_grid, 
    calculate_potential_bound, classify_regime,
    coalition_structure_metrics, strategy_profile_metrics,
    ExperimentLogger, coalition_structure_to_string
)


class ExperimentSuite:
    """Suite of experiments for coalitional congestion games."""
    
    def __init__(
        self,
        n_players: int = 6,
        resource_type: str = 'heterogeneous',
        output_dir: str = 'results',
        verbose: bool = True
    ):
        """
        Initialize experiment suite.
        
        Args:
            n_players: Number of players
            resource_type: Type of resources ('heterogeneous', 'homogeneous', 'random')
            output_dir: Directory for results
            verbose: Whether to print progress
        """
        self.n_players = n_players
        self.resources = create_standard_resources(resource_type)
        self.logger = ExperimentLogger(output_dir)
        self.verbose = verbose
        
        # Calculate theoretical bound
        self.potential_bound = calculate_potential_bound(self.resources, n_players)
        
        if self.verbose:
            print(f"Initialized experiment suite:")
            print(f"  Players: {n_players}")
            print(f"  Resources: {len(self.resources)} ({resource_type})")
            print(f"  Potential-compatibility bound: {self.potential_bound:.6f}")
    
    def experiment_1_archetype_stability_phases(
        self,
        parameter_grid: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Experiment 1: Stability Phase Diagrams of Three Archetype Structures.
        
        Tests whether three canonical coalition structures can "survive" 
        (remain stable) across the (rho, sigma) parameter space:
        
        1. Grand Coalition: All players together
        2. Singleton: Complete fragmentation  
        3. Factions: Two equal opposing groups
        
        For each (rho, sigma) and each archetype:
        - Step A: Compute Nash equilibrium given the structure
        - Step B: Check if any profitable split/join exists
        - Mark as "Stable" if no profitable deviation found
        
        Expected pattern validates Section 3.6 regimes:
        - Grand stable in high-ρ low-σ (altruistic)
        - Singleton stable in low-ρ high-σ (spiteful)
        - Factions stable in high-ρ high-σ (factional)
        
        Args:
            parameter_grid: List of (rho, sigma) pairs to test
        """
        from stability import StabilityChecker
        
        # Clear previous results
        self.logger.results = []
        
        if parameter_grid is None:
            parameter_grid = create_parameter_grid()
        
        # Define three archetype structures
        archetypes = {
            'grand': [set(range(self.n_players))],
            'singleton': [{i} for i in range(self.n_players)],
            'factions': [
                set(range(self.n_players // 2)),
                set(range(self.n_players // 2, self.n_players))
            ]
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EXPERIMENT 1: Archetype Stability Phase Diagrams")
            print(f"{'='*60}")
            print(f"Testing 3 archetype structures across {len(parameter_grid)} parameter points")
            print(f"Archetypes: Grand Coalition, Singleton, Two-Faction")
        
        results = []
        
        for rho, sigma in tqdm(parameter_grid, desc="Parameters", disable=not self.verbose):
            regime = classify_regime(rho, sigma, self.potential_bound)
            
            for archetype_name, structure in archetypes.items():
                # Create game with this fixed structure
                game = CongestionGame(
                    n_players=self.n_players,
                    resources=self.resources,
                    coalition_structure=structure,
                    rho=rho,
                    sigma=sigma
                )
                
                # Step A: Find Nash equilibrium
                solver = EquilibriumSolver(game, max_iterations=500)
                profile, converged, iterations, _ = solver.best_response_dynamics(verbose=False)
                
                if not converged:
                    # Cannot assess stability without equilibrium
                    result = {
                        'rho': rho,
                        'sigma': sigma,
                        'regime': regime,
                        'archetype': archetype_name,
                        'converged': False,
                        'is_stable': False,
                        'split_stable': False,
                        'join_stable': False,
                    }
                    self.logger.log_result(result)
                    results.append(result)
                    continue
                
                # Step B: Check structural stability
                checker = StabilityChecker(game)
                split_stable, split_info = checker.test_split_stability(profile, verbose=False)
                join_stable, join_info = checker.test_join_stability(profile, verbose=False)
                
                is_stable = split_stable and join_stable
                
                result = {
                    'rho': rho,
                    'sigma': sigma,
                    'regime': regime,
                    'archetype': archetype_name,
                    'converged': True,
                    'convergence_iterations': iterations,
                    'is_stable': is_stable,
                    'split_stable': split_stable,
                    'join_stable': join_stable,
                    'social_cost': float(game.social_cost(profile)),
                    'rosenthal_potential': float(game.rosenthal_potential(profile)),
                }
                
                self.logger.log_result(result)
                results.append(result)
        
        if self.verbose:
            print(f"\nCompleted {len(results)} tests")
            print("\nStability rates by archetype:")
            for arch in archetypes.keys():
                arch_results = [r for r in results if r['archetype'] == arch and r['converged']]
                if arch_results:
                    stable_rate = sum(r['is_stable'] for r in arch_results) / len(arch_results)
                    print(f"  {arch}: {stable_rate:.1%} stable")
        
        self.logger.save_results('experiment_1_archetype_stability.json')
        return results
    
    def experiment_2_splitting_mechanism(
        self,
        fixed_rho: float = 0.0005,
        sigma_values: Optional[List[float]] = None
    ):
        """
        Experiment 2: Mechanism of Splitting - The Spite Incentive.
        
        Explains WHY increasing σ causes grand coalition to collapse.
        Shows that splitting is NOT about efficiency loss, but about gaining
        "attack capability" through out-group hostility.
        
        Setup:
        - Fix structure as Grand Coalition
        - Fix ρ = 0.0005 (moderate altruism)
        - Sweep σ from 0 to 0.01
        
        Measures:
        - Split Incentive: Δ V = max(U_split_groups) - U_grand
        - Resource choices before/after split
        - Demonstrates weaponization: splitters choose congested resources
          to harm remainers, sacrificing own μ but gaining through σ term
        
        Args:
            fixed_rho: Fixed altruism level
            sigma_values: Range of spite levels to test
        """
        from stability import StabilityChecker
        
        # Clear previous results
        self.logger.results = []
        
        if sigma_values is None:
            sigma_values = np.linspace(0, 0.01, 16).tolist()
        
        # Grand coalition structure
        grand_structure = [set(range(self.n_players))]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EXPERIMENT 2: Splitting Mechanism (Spite Incentive)")
            print(f"{'='*60}")
            print(f"Fixed ρ = {fixed_rho}")
            print(f"Testing σ ∈ [{min(sigma_values):.2f}, {max(sigma_values):.2f}]")
            print(f"Structure: Grand Coalition")
        
        results = []
        
        for sigma in tqdm(sigma_values, desc="Sigma values", disable=not self.verbose):
            # Create game with grand coalition
            game = CongestionGame(
                n_players=self.n_players,
                resources=self.resources,
                coalition_structure=grand_structure,
                rho=fixed_rho,
                sigma=sigma
            )
            
            # Find Nash equilibrium
            solver = EquilibriumSolver(game, max_iterations=500)
            profile_grand, converged, _, _ = solver.best_response_dynamics(verbose=False)
            
            if not converged:
                continue
            
            # Calculate utility of grand coalition
            U_grand = game.get_coalition_utility(grand_structure[0], profile_grand)
            
            # Find most profitable split
            checker = StabilityChecker(game)
            split_stable, split_info = checker.test_split_stability(profile_grand, verbose=False)
            
            if split_stable:
                # No profitable split
                max_split_incentive = 0.0
                split_gain = 0.0
            else:
                # Calculate split incentive
                split_a = split_info['split_a']
                split_b = split_info['split_b']
                U_split_total = split_info['new_utility_a'] + split_info['new_utility_b']
                split_gain = split_info['gain']
                max_split_incentive = U_split_total - U_grand
                
                # Analyze resource choices after split
                new_structure = [split_a, split_b]
                game_split = game.copy_with_new_coalition_structure(new_structure)
                solver_split = EquilibriumSolver(game_split, max_iterations=500)
                profile_split, converged_split, _, _ = solver_split.best_response_dynamics(verbose=False)
                
                if converged_split:
                    congestion_split = game_split.get_congestion(profile_split)
                else:
                    congestion_split = None
            
            # Record result
            result = {
                'rho': fixed_rho,
                'sigma': float(sigma),
                'regime': classify_regime(fixed_rho, sigma, self.potential_bound),
                'U_grand': float(U_grand),
                'is_split_stable': split_stable,
                'split_incentive': float(max_split_incentive),
                'split_gain': float(split_gain),
                'social_cost_grand': float(game.social_cost(profile_grand)),
            }
            
            self.logger.log_result(result)
            results.append(result)
        
        if self.verbose:
            print(f"\nCompleted {len(results)} sigma values")
            print("\nKey transitions:")
            for i, r in enumerate(results):
                if i > 0 and results[i-1]['is_split_stable'] and not r['is_split_stable']:
                    print(f"  Transition to unstable at σ ≈ {r['sigma']:.3f}")
                    print(f"  Split incentive: {r['split_incentive']:.3f}")
        
        self.logger.save_results('experiment_2_splitting_mechanism.json')
        return results
    
    def experiment_3_price_of_stability(
        self,
        parameter_grid: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Experiment 3: The Price of Stability (Efficiency Analysis).
        
        While we focus on stability, we must assess EFFICIENCY:
        If social preferences induce a stable coalition structure, is it
        good or bad for the transportation system?
        
        For each (ρ, σ):
        1. Identify the most stable archetype (from Exp 1 results)
        2. Compute social cost at that structure's equilibrium
        3. Calculate PoA-like ratio: Cost(social pref) / Cost(selfish NE)
        
        Expected patterns:
        - Altruistic region: Cost decreases (Ratio < 1)
          Grand coalition coordinates, eliminates externalities
        - Spiteful region: Cost explodes (Ratio > 1)  
          Weaponized congestion: players deliberately create jams
          to harm enemies
        - Nearly selfish: Cost ≈ baseline
        
        Key insight: Social preferences don't always help!
        Spite-induced stability can be catastrophic for efficiency.
        
        Args:
            parameter_grid: List of (rho, sigma) pairs
        """
        from stability import StabilityChecker
        
        # Clear previous results
        self.logger.results = []
        
        if parameter_grid is None:
            parameter_grid = create_parameter_grid()
        
        # Define archetypes
        archetypes = {
            'grand': [set(range(self.n_players))],
            'singleton': [{i} for i in range(self.n_players)],
            'factions': [
                set(range(self.n_players // 2)),
                set(range(self.n_players // 2, self.n_players))
            ]
        }
        
        # Compute selfish baseline once
        selfish_game = CongestionGame(
            n_players=self.n_players,
            resources=self.resources,
            coalition_structure=[{i} for i in range(self.n_players)],  # Singleton
            rho=0.0,
            sigma=0.0
        )
        solver_selfish = EquilibriumSolver(selfish_game, max_iterations=500)
        profile_selfish, _, _, _ = solver_selfish.best_response_dynamics(verbose=False)
        baseline_cost = selfish_game.social_cost(profile_selfish)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EXPERIMENT 3: Price of Stability (Efficiency Analysis)")
            print(f"{'='*60}")
            print(f"Baseline selfish cost: {baseline_cost:.2f}")
            print(f"Testing {len(parameter_grid)} parameter points")
        
        results = []
        
        for rho, sigma in tqdm(parameter_grid, desc="Parameters", disable=not self.verbose):
            regime = classify_regime(rho, sigma, self.potential_bound)
            
            # Test all archetypes, find most stable
            archetype_scores = {}
            
            for arch_name, structure in archetypes.items():
                game = CongestionGame(
                    n_players=self.n_players,
                    resources=self.resources,
                    coalition_structure=structure,
                    rho=rho,
                    sigma=sigma
                )
                
                solver = EquilibriumSolver(game, max_iterations=500)
                profile, converged, _, _ = solver.best_response_dynamics(verbose=False)
                
                if not converged:
                    archetype_scores[arch_name] = {
                        'stable': False,
                        'cost': float('inf')
                    }
                    continue
                
                # Check stability
                checker = StabilityChecker(game)
                split_stable, _ = checker.test_split_stability(profile, verbose=False)
                join_stable, _ = checker.test_join_stability(profile, verbose=False)
                is_stable = split_stable and join_stable
                
                social_cost = game.social_cost(profile)
                
                archetype_scores[arch_name] = {
                    'stable': is_stable,
                    'cost': float(social_cost),
                    'profile': profile
                }
            
            # Select most stable archetype (prioritize stability, then cost)
            stable_archetypes = [(name, info) for name, info in archetype_scores.items() 
                                if info['stable']]
            
            if stable_archetypes:
                # Pick stable one with lowest cost
                best_arch_name, best_info = min(stable_archetypes, 
                                               key=lambda x: x[1]['cost'])
            else:
                # No stable structure, pick least unstable (lowest cost)
                best_arch_name, best_info = min(archetype_scores.items(),
                                               key=lambda x: x[1]['cost'])
            
            # Calculate PoA-like ratio
            cost_ratio = best_info['cost'] / baseline_cost
            
            result = {
                'rho': rho,
                'sigma': sigma,
                'regime': regime,
                'best_archetype': best_arch_name,
                'is_stable': best_info['stable'],
                'social_cost': best_info['cost'],
                'baseline_cost': float(baseline_cost),
                'cost_ratio': float(cost_ratio),
                'improvement': float(1.0 - cost_ratio),  # Positive = better than selfish
            }
            
            self.logger.log_result(result)
            results.append(result)
        
        if self.verbose:
            print(f"\nCompleted {len(results)} parameter points")
            print("\nEfficiency by regime:")
            for regime_name in ['nearly_selfish', 'altruistic', 'spiteful', 'factional']:
                regime_results = [r for r in results if r['regime'] == regime_name]
                if regime_results:
                    avg_ratio = np.mean([r['cost_ratio'] for r in regime_results])
                    print(f"  {regime_name}: avg cost ratio = {avg_ratio:.3f}")
        
        self.logger.save_results('experiment_3_price_of_stability.json')
        return results
    
    def experiment_4_fragility_of_generic_structures(
        self,
        n_random_structures: int = 100,
        test_rho: float = 0.0002,
        test_sigma: float = 0.0002
    ):
        """
        Experiment 4: Fragility of Generic Coalition Structures.
        
        Demonstrates that arbitrary/asymmetric coalition structures are
        almost never stable. Only highly symmetric or extreme structures
        (Grand, Singleton, Factions) can robustly maintain stability.
        
        This validates WHY Experiment 1 focuses only on three archetypes:
        because generic structures are inherently fragile.
        
        Procedure:
        1. Generate 100 random asymmetric structures 
           (e.g., {1}, {2,3,4}, {5,6})
        2. Test at moderate parameters (ρ=0.0002, σ=0.0002)
        3. Check structural stability (split + join)
        4. Record failure patterns
        
        Expected result:
        - >95% unstable
        - Most common failure: profitable splits or joins exist
        - Shows that stability requires extreme symmetry
        
        Args:
            n_random_structures: Number of random structures to generate
            test_rho: Altruism level for testing
            test_sigma: Spite level for testing
        """
        from stability import sample_random_coalition_structures, StabilityChecker
        
        # Clear previous results
        self.logger.results = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("EXPERIMENT 4: Fragility of Generic Structures")
            print(f"{'='*60}")
            print(f"Testing {n_random_structures} random structures")
            print(f"At moderate parameters: ρ={test_rho}, σ={test_sigma}")
        
        # Generate random asymmetric structures
        random_structures = sample_random_coalition_structures(
            self.n_players, n_random_structures, seed=42
        )
        
        # Also include the three archetypes for comparison
        archetypes = {
            'grand': [set(range(self.n_players))],
            'singleton': [{i} for i in range(self.n_players)],
            'factions': [
                set(range(self.n_players // 2)),
                set(range(self.n_players // 2, self.n_players))
            ]
        }
        
        results = []
        failure_reasons = {'split': 0, 'join': 0, 'both': 0, 'stable': 0}
        
        # Test archetypes first
        for arch_name, structure in archetypes.items():
            game = CongestionGame(
                n_players=self.n_players,
                resources=self.resources,
                coalition_structure=structure,
                rho=test_rho,
                sigma=test_sigma
            )
            
            solver = EquilibriumSolver(game, max_iterations=500)
            profile, converged, _, _ = solver.best_response_dynamics(verbose=False)
            
            if not converged:
                continue
            
            checker = StabilityChecker(game)
            split_stable, _ = checker.test_split_stability(profile, verbose=False)
            join_stable, _ = checker.test_join_stability(profile, verbose=False)
            is_stable = split_stable and join_stable
            
            # Categorize failure
            if is_stable:
                failure_type = 'stable'
                failure_reasons['stable'] += 1
            elif not split_stable and not join_stable:
                failure_type = 'both'
                failure_reasons['both'] += 1
            elif not split_stable:
                failure_type = 'split'
                failure_reasons['split'] += 1
            else:
                failure_type = 'join'
                failure_reasons['join'] += 1
            
            result = {
                'structure_type': 'archetype',
                'structure_name': arch_name,
                'n_coalitions': len(structure),
                'converged': True,
                'is_stable': is_stable,
                'split_stable': split_stable,
                'join_stable': join_stable,
                'failure_type': failure_type,
            }
            
            results.append(result)
        
        # Test random structures
        for idx, structure in enumerate(tqdm(random_structures, desc="Testing", disable=not self.verbose)):
            game = CongestionGame(
                n_players=self.n_players,
                resources=self.resources,
                coalition_structure=structure,
                rho=test_rho,
                sigma=test_sigma
            )
            
            solver = EquilibriumSolver(game, max_iterations=500)
            profile, converged, _, _ = solver.best_response_dynamics(verbose=False)
            
            if not converged:
                result = {
                    'structure_type': 'random',
                    'structure_id': idx,
                    'n_coalitions': len(structure),
                    'converged': False,
                    'is_stable': False,
                    'split_stable': False,
                    'join_stable': False,
                    'failure_type': 'no_equilibrium',
                }
                results.append(result)
                continue
            
            checker = StabilityChecker(game)
            split_stable, _ = checker.test_split_stability(profile, verbose=False)
            join_stable, _ = checker.test_join_stability(profile, verbose=False)
            is_stable = split_stable and join_stable
            
            # Categorize failure
            if is_stable:
                failure_type = 'stable'
                failure_reasons['stable'] += 1
            elif not split_stable and not join_stable:
                failure_type = 'both'
                failure_reasons['both'] += 1
            elif not split_stable:
                failure_type = 'split'
                failure_reasons['split'] += 1
            else:
                failure_type = 'join'
                failure_reasons['join'] += 1
            
            # Calculate structure asymmetry (variance in coalition sizes)
            sizes = [len(c) for c in structure]
            size_variance = float(np.var(sizes))
            
            result = {
                'structure_type': 'random',
                'structure_id': idx,
                'n_coalitions': len(structure),
                'size_variance': size_variance,
                'converged': True,
                'is_stable': is_stable,
                'split_stable': split_stable,
                'join_stable': join_stable,
                'failure_type': failure_type,
            }
            
            results.append(result)
        
        # Save all results
        for r in results:
            r['rho'] = test_rho
            r['sigma'] = test_sigma
            self.logger.log_result(r)
        
        if self.verbose:
            print(f"\nCompleted {len(results)} structures")
            
            # Statistics
            random_results = [r for r in results if r['structure_type'] == 'random' and r['converged']]
            arch_results = [r for r in results if r['structure_type'] == 'archetype']
            
            if random_results:
                stable_rate = sum(r['is_stable'] for r in random_results) / len(random_results)
                print(f"\nRandom structures:")
                print(f"  Stable: {stable_rate:.1%}")
                print(f"  Tested: {len(random_results)}")
            
            if arch_results:
                arch_stable = sum(r['is_stable'] for r in arch_results) / len(arch_results)
                print(f"\nArchetypes:")
                print(f"  Stable: {arch_stable:.1%}")
            
            print(f"\nFailure breakdown:")
            total_failures = sum(failure_reasons.values())
            for reason, count in failure_reasons.items():
                pct = count / total_failures if total_failures > 0 else 0
                print(f"  {reason}: {count} ({pct:.1%})")
        
        self.logger.save_results('experiment_4_structure_fragility.json')
        return results
    
    def run_all_experiments(self):
        """Run all experiments in sequence."""
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING COMPLETE EXPERIMENT SUITE (STATIC STABILITY ANALYSIS)")
            print("="*70)
            print("\n[1] Archetype Stability Phases: Which structures survive where?")
            print("[2] Splitting Mechanism: Why does σ break grand coalitions?")
            print("[3] Price of Stability: Efficiency of stable structures")
            print("[4] Structure Fragility: Why only extremes are stable?\n")
        
        # Run experiments
        self.experiment_1_archetype_stability_phases()
        self.experiment_2_splitting_mechanism()
        self.experiment_3_price_of_stability()
        self.experiment_4_fragility_of_generic_structures()
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ALL EXPERIMENTS COMPLETED")
            print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            print(f"{'='*70}")


def quick_test():
    """Quick test to verify all components work."""
    print("Running quick test...")
    
    # Create small game
    n_players = 4
    resources = create_standard_resources('heterogeneous')
    structure = [{0, 1}, {2, 3}]  # Two pairs
    
    game = CongestionGame(
        n_players=n_players,
        resources=resources,
        coalition_structure=structure,
        rho=0.2,
        sigma=0.1
    )
    
    # Test equilibrium solver
    solver = EquilibriumSolver(game, max_iterations=100)
    profile, converged, iterations, _ = solver.best_response_dynamics(verbose=True)
    
    print(f"\nConverged: {converged} in {iterations} iterations")
    print(f"Final profile: {profile}")
    print(f"Social cost: {game.social_cost(profile):.2f}")
    
    # Test stability
    checker = StabilityChecker(game)
    is_eq, reason, info = checker.is_coalition_equilibrium(profile, verbose=True)
    
    print(f"\nCoalition equilibrium: {is_eq}")
    if not is_eq:
        print(f"Reason: {reason}")
    
    print("\nQuick test completed successfully!")


if __name__ == '__main__':
    # Run quick test
    quick_test()
    
    # Uncomment to run full experiments
    # suite = ExperimentSuite(n_players=6, resource_type='heterogeneous')
    # suite.run_all_experiments()

