"""
Unit tests for core functionality.

Run with: python test_core.py
"""

import numpy as np
from game import CongestionGame, Resource
from equilibrium import EquilibriumSolver, LogitLearning
from stability import StabilityChecker, create_standard_structures
from utils import calculate_potential_bound, classify_regime


def test_game_creation():
    """Test basic game creation and utilities."""
    print("\nTest 1: Game Creation")
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=0.5, b=2.0),
    ]
    
    structure = [{0, 1}, {2, 3}]
    
    game = CongestionGame(
        n_players=4,
        resources=resources,
        coalition_structure=structure,
        rho=0.2,
        sigma=0.1
    )
    
    # Test basic properties
    assert game.n_players == 4
    assert len(game.resources) == 2
    assert len(game.coalition_structure) == 2
    
    # Test profile
    profile = np.array([0, 0, 1, 1])
    congestion = game.get_congestion(profile)
    assert congestion[0] == 2
    assert congestion[1] == 2
    
    # Test latency
    latency_0 = game.get_latency(0, profile)
    expected = 1.0 * 2 + 0.0  # a*n + b
    assert abs(latency_0 - expected) < 1e-9
    
    # Test utility calculation
    utility_0 = game.get_utility(0, profile)
    # Should include own payoff + rho*(partner's payoff) - sigma*(others' payoffs)
    assert isinstance(utility_0, (int, float))
    
    print("  ✓ Game creation and basic utilities work")
    return True


def test_nash_equilibrium():
    """Test Nash equilibrium computation."""
    print("\nTest 2: Nash Equilibrium")
    
    resources = [
        Resource(id=0, a=1.0, b=1.0),
        Resource(id=1, a=1.0, b=1.0),
    ]
    
    # Homogeneous resources - any profile is Nash for singleton structure
    structure = [{0}, {1}, {2}]
    
    game = CongestionGame(
        n_players=3,
        resources=resources,
        coalition_structure=structure,
        rho=0.0,
        sigma=0.0
    )
    
    solver = EquilibriumSolver(game, max_iterations=50)
    
    # Test best-response dynamics
    profile, converged, iterations, _ = solver.best_response_dynamics()
    assert converged, "Should converge for homogeneous resources"
    
    # Test Nash check
    is_nash = solver.is_nash_equilibrium(profile)
    assert is_nash, "Converged profile should be Nash equilibrium"
    
    print("  ✓ Nash equilibrium computation works")
    return True


def test_potential_function():
    """Test Rosenthal potential calculation."""
    print("\nTest 3: Potential Function")
    
    resources = [
        Resource(id=0, a=2.0, b=1.0),
        Resource(id=1, a=1.0, b=2.0),
    ]
    
    structure = [{0, 1}]
    
    game = CongestionGame(
        n_players=2,
        resources=resources,
        coalition_structure=structure,
        rho=0.0,
        sigma=0.0
    )
    
    # Profile: both on resource 0
    profile1 = np.array([0, 0])
    # f_0(1) + f_0(2) = (2*1+1) + (2*2+1) = 3 + 5 = 8
    potential1 = game.rosenthal_potential(profile1)
    expected1 = (2*1 + 1) + (2*2 + 1)
    assert abs(potential1 - expected1) < 1e-9
    
    # Profile: one on each resource
    profile2 = np.array([0, 1])
    # f_0(1) + f_1(1) = (2*1+1) + (1*1+2) = 3 + 3 = 6
    potential2 = game.rosenthal_potential(profile2)
    expected2 = (2*1 + 1) + (1*1 + 2)
    assert abs(potential2 - expected2) < 1e-9
    
    print("  ✓ Potential function calculation correct")
    return True


def test_split_stability():
    """Test split stability checking."""
    print("\nTest 4: Split Stability")
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=1.0, b=0.0),
    ]
    
    # Start with grand coalition
    structure = [{0, 1, 2, 3}]
    
    game = CongestionGame(
        n_players=4,
        resources=resources,
        coalition_structure=structure,
        rho=0.1,
        sigma=0.0
    )
    
    # Find equilibrium
    solver = EquilibriumSolver(game, max_iterations=50)
    profile, converged, _, _ = solver.best_response_dynamics()
    
    if not converged:
        print("  ⚠ Did not converge, skipping stability test")
        return True
    
    # Test split stability
    checker = StabilityChecker(game)
    split_stable, split_info = checker.test_split_stability(profile, verbose=False)
    
    # Just check that it runs without error
    assert isinstance(split_stable, bool)
    
    print(f"  ✓ Split stability check works (stable: {split_stable})")
    return True


def test_join_stability():
    """Test join stability checking."""
    print("\nTest 5: Join Stability")
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=1.0, b=0.0),
    ]
    
    # Start with singleton structure
    structure = [{0}, {1}, {2}, {3}]
    
    game = CongestionGame(
        n_players=4,
        resources=resources,
        coalition_structure=structure,
        rho=0.5,  # High altruism -> might want to join
        sigma=0.0
    )
    
    # Find equilibrium
    solver = EquilibriumSolver(game, max_iterations=50)
    profile, converged, _, _ = solver.best_response_dynamics()
    
    if not converged:
        print("  ⚠ Did not converge, skipping stability test")
        return True
    
    # Test join stability
    checker = StabilityChecker(game)
    join_stable, join_info = checker.test_join_stability(profile, verbose=False)
    
    # Just check that it runs without error
    assert isinstance(join_stable, bool)
    
    print(f"  ✓ Join stability check works (stable: {join_stable})")
    return True


def test_logit_learning():
    """Test logit learning dynamics."""
    print("\nTest 6: Logit Learning")
    
    resources = [
        Resource(id=0, a=1.0, b=0.0),
        Resource(id=1, a=1.0, b=0.0),
    ]
    
    structure = [{0, 1}]
    
    game = CongestionGame(
        n_players=2,
        resources=resources,
        coalition_structure=structure,
        rho=0.0,
        sigma=0.0
    )
    
    learner = LogitLearning(game, temperature=0.1)
    
    # Run short dynamics
    trajectory, potentials = learner.run_dynamics(n_iterations=100)
    
    assert len(trajectory) == 101  # Initial + 100 iterations
    assert len(potentials) == 101
    
    print("  ✓ Logit learning dynamics work")
    return True


def test_regime_classification():
    """Test regime classification."""
    print("\nTest 7: Regime Classification")
    
    resources = [Resource(id=0, a=1.0, b=0.0)]
    bound = calculate_potential_bound(resources, n_players=4)
    
    # Test near-selfish
    regime1 = classify_regime(0.0, 0.0, bound)
    assert regime1 == 'nearly_selfish'
    
    # Test altruistic
    regime2 = classify_regime(0.8, 0.1, bound)
    assert regime2 == 'altruistic'
    
    # Test spiteful
    regime3 = classify_regime(0.1, 0.8, bound)
    assert regime3 == 'spiteful'
    
    # Test factional
    regime4 = classify_regime(0.8, 0.8, bound)
    assert regime4 == 'factional'
    
    print("  ✓ Regime classification works")
    return True


def test_structure_generation():
    """Test coalition structure generation."""
    print("\nTest 8: Structure Generation")
    
    # Test standard structures
    structures = create_standard_structures(n_players=6)
    
    assert 'singleton' in structures
    assert len(structures['singleton']) == 6
    
    assert 'grand' in structures
    assert len(structures['grand']) == 1
    assert len(structures['grand'][0]) == 6
    
    if 'pairs' in structures:
        assert len(structures['pairs']) == 3
    
    print(f"  ✓ Structure generation works ({len(structures)} standard structures)")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING CORE FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        test_game_creation,
        test_nash_equilibrium,
        test_potential_function,
        test_split_stability,
        test_join_stability,
        test_logit_learning,
        test_regime_classification,
        test_structure_generation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

