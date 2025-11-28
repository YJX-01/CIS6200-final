"""
Example usage demonstrating the basic workflow.

This script walks through:
1. Creating a congestion game
2. Finding Nash equilibria
3. Testing coalition stability
4. Running learning dynamics
"""

import numpy as np
from game import CongestionGame, Resource
from equilibrium import EquilibriumSolver, LogitLearning
from stability import StabilityChecker, create_standard_structures
from utils import pretty_print_game_state, calculate_potential_bound, classify_regime


def example_1_basic_game():
    """Example 1: Create a basic game and find equilibrium."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Congestion Game")
    print("="*60)
    
    # Create resources (from paper)
    resources = [
        Resource(id=0, a=1.0, b=0.0),   # Tight: high congestion sensitivity
        Resource(id=1, a=0.5, b=2.0),   # Forgiving: low sensitivity
        Resource(id=2, a=1.5, b=1.0),   # Medium
    ]
    
    # Create game with 6 players in pairs
    coalition_structure = [{0, 1}, {2, 3}, {4, 5}]
    
    game = CongestionGame(
        n_players=6,
        resources=resources,
        coalition_structure=coalition_structure,
        rho=0.2,    # Moderate in-group altruism
        sigma=0.1   # Low out-group spite
    )
    
    print(f"\nGame configuration:")
    print(f"  Players: {game.n_players}")
    print(f"  Resources: {len(resources)}")
    print(f"  Coalition structure: {coalition_structure}")
    print(f"  Social preferences: ρ={game.rho}, σ={game.sigma}")
    
    # Calculate theoretical bound
    bound = calculate_potential_bound(resources, game.n_players)
    regime = classify_regime(game.rho, game.sigma, bound)
    print(f"  Potential-compatibility bound: {bound:.6f}")
    print(f"  Regime: {regime}")
    
    # Find Nash equilibrium
    print("\nFinding Nash equilibrium via best-response dynamics...")
    solver = EquilibriumSolver(game, max_iterations=100)
    profile, converged, iterations, potential_trajectory = \
        solver.best_response_dynamics(verbose=False)
    
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Strategy profile: {profile}")
    
    # Display results
    pretty_print_game_state(game, profile, title="Nash Equilibrium")
    
    return game, profile


def example_2_stability_testing(game, profile):
    """Example 2: Test coalition stability."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Coalition Stability Testing")
    print("="*60)
    
    checker = StabilityChecker(game)
    
    # Test split stability
    print("\nTesting split stability...")
    split_stable, split_info = checker.test_split_stability(profile, verbose=True)
    
    if split_stable:
        print("✓ Split stable: No coalition wants to split")
    else:
        print("✗ Split unstable: Found profitable split")
        print(f"  Gain: {split_info['gain']:.4f}")
    
    # Test join stability
    print("\nTesting join stability...")
    join_stable, join_info = checker.test_join_stability(profile, verbose=True)
    
    if join_stable:
        print("✓ Join stable: No coalitions want to merge")
    else:
        print("✗ Join unstable: Found profitable join")
        print(f"  Gain: {join_info['gain']:.4f}")
    
    # Overall coalition equilibrium check
    print("\nChecking coalition equilibrium...")
    is_eq, reason, info = checker.is_coalition_equilibrium(profile, verbose=False)
    
    if is_eq:
        print("✓ This is a COALITION EQUILIBRIUM!")
    else:
        print(f"✗ Not a coalition equilibrium: {reason}")
    
    return is_eq


def example_3_learning_dynamics(game):
    """Example 3: Run logit learning dynamics."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Logit Learning Dynamics")
    print("="*60)
    
    # Run with low temperature (high rationality)
    print("\nRunning logit learning with τ=0.1...")
    learner = LogitLearning(game, temperature=0.1)
    trajectory, potentials = learner.run_dynamics(n_iterations=1000)
    
    print(f"  Ran {len(trajectory)} iterations")
    print(f"  Initial potential: {potentials[0]:.2f}")
    print(f"  Final potential: {potentials[-1]:.2f}")
    print(f"  Potential change: {potentials[-1] - potentials[0]:.2f}")
    
    # Analyze final state
    final_profile = trajectory[-1]
    print(f"  Final strategy profile: {final_profile}")
    print(f"  Final social cost: {game.social_cost(final_profile):.2f}")
    
    # Check if final state is Nash equilibrium
    solver = EquilibriumSolver(game)
    is_nash = solver.is_nash_equilibrium(final_profile)
    print(f"  Is Nash equilibrium: {is_nash}")
    
    return trajectory


def example_4_regime_comparison():
    """Example 4: Compare different social preference regimes."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Social Preference Regimes")
    print("="*60)
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=0.5, b=2.0),
        Resource(id=2, a=1.5, b=1.0),
    ]
    
    # Test different regimes
    regimes = [
        ("Nearly Selfish", 0.0, 0.0),
        ("Altruistic", 0.8, 0.1),
        ("Spiteful", 0.1, 0.8),
        ("Factional", 0.6, 0.6),
    ]
    
    # Use pairs coalition structure
    coalition_structure = [{0, 1}, {2, 3}, {4, 5}]
    
    results = []
    
    for regime_name, rho, sigma in regimes:
        print(f"\n--- {regime_name} Regime (ρ={rho}, σ={sigma}) ---")
        
        game = CongestionGame(
            n_players=6,
            resources=resources,
            coalition_structure=coalition_structure,
            rho=rho,
            sigma=sigma
        )
        
        # Find equilibrium
        solver = EquilibriumSolver(game, max_iterations=100)
        profile, converged, iterations, _ = solver.best_response_dynamics(verbose=False)
        
        if not converged:
            print("  Failed to converge")
            continue
        
        # Test stability
        checker = StabilityChecker(game)
        split_stable, _ = checker.test_split_stability(profile)
        join_stable, _ = checker.test_join_stability(profile)
        is_eq = split_stable and join_stable
        
        # Metrics
        social_cost = game.social_cost(profile)
        congestion = game.get_congestion(profile)
        
        print(f"  Converged in {iterations} iterations")
        print(f"  Strategy: {profile}")
        print(f"  Social cost: {social_cost:.2f}")
        print(f"  Congestion: {congestion}")
        print(f"  Split stable: {split_stable}")
        print(f"  Join stable: {join_stable}")
        print(f"  Coalition equilibrium: {is_eq}")
        
        results.append({
            'regime': regime_name,
            'rho': rho,
            'sigma': sigma,
            'converged': converged,
            'social_cost': social_cost,
            'is_coalition_eq': is_eq
        })
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"{'Regime':<20} {'ρ':<6} {'σ':<6} {'Cost':<8} {'Stable'}")
    print("-"*60)
    for r in results:
        print(f"{r['regime']:<20} {r['rho']:<6.1f} {r['sigma']:<6.1f} "
              f"{r['social_cost']:<8.2f} {r['is_coalition_eq']}")


def example_5_structure_comparison():
    """Example 5: Compare different coalition structures."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Coalition Structure Comparison")
    print("="*60)
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=0.5, b=2.0),
        Resource(id=2, a=1.5, b=1.0),
    ]
    
    # Fixed social preferences
    rho, sigma = 0.3, 0.2
    
    # Different structures
    structures = create_standard_structures(n_players=6)
    
    print(f"\nTesting structures with ρ={rho}, σ={sigma}")
    
    results = []
    
    for name, structure in structures.items():
        print(f"\n--- {name} ---")
        print(f"  Structure: {structure}")
        
        game = CongestionGame(
            n_players=6,
            resources=resources,
            coalition_structure=structure,
            rho=rho,
            sigma=sigma
        )
        
        # Find equilibrium
        solver = EquilibriumSolver(game, max_iterations=100)
        profile, converged, iterations, _ = solver.best_response_dynamics(verbose=False)
        
        if not converged:
            print("  Failed to converge")
            continue
        
        # Test stability
        checker = StabilityChecker(game)
        is_eq, reason, _ = checker.is_coalition_equilibrium(profile, verbose=False)
        
        social_cost = game.social_cost(profile)
        
        print(f"  Converged in {iterations} iterations")
        print(f"  Social cost: {social_cost:.2f}")
        print(f"  Coalition equilibrium: {is_eq}")
        if not is_eq:
            print(f"  Failure reason: {reason}")
        
        results.append({
            'name': name,
            'n_coalitions': len(structure),
            'social_cost': social_cost,
            'is_eq': is_eq
        })
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"{'Structure':<20} {'#Coalitions':<15} {'Cost':<10} {'Stable'}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['n_coalitions']:<15} "
              f"{r['social_cost']:<10.2f} {r['is_eq']}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "COALITIONAL CONGESTION GAMES")
    print(" "*20 + "Example Usage Demo")
    print("="*70)
    
    # Example 1: Basic game
    game, profile = example_1_basic_game()
    
    # Example 2: Stability testing
    example_2_stability_testing(game, profile)
    
    # Example 3: Learning dynamics
    example_3_learning_dynamics(game)
    
    # Example 4: Regime comparison
    example_4_regime_comparison()
    
    # Example 5: Structure comparison
    example_5_structure_comparison()
    
    print("\n" + "="*70)
    print(" "*20 + "All examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

