# Coalitional Congestion Games with Social Preferences

Experimental codebase for studying coalition formation and stability in congestion games with in-group altruism and out-group spite.

## Overview

This codebase implements the theoretical framework described in the paper "Coalitional Stability in Congestion Games with Altruism and Spite". It provides tools for:

- **Congestion game modeling** with linear latency functions
- **Social preferences**: In-group altruism (ρ) and out-group spite (σ)
- **Nash equilibrium computation** via best-response dynamics
- **Coalition stability analysis** (split/join deviations)
- **Learning dynamics** with logit response
- **Comprehensive experiments** across parameter space

## Project Structure

```
new/
├── game.py              # Core congestion game class
├── equilibrium.py       # Nash equilibrium solvers
├── stability.py         # Coalition stability testing
├── utils.py             # Utility functions and metrics
├── experiments.py       # Experiment framework
├── visualization.py     # Plotting and visualization
├── run_experiments.py   # Main execution script
└── README.md           # This file
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- tqdm

### Setup

```bash
# Install dependencies
pip install numpy matplotlib seaborn tqdm

# Or use requirements.txt (if provided)
pip install -r requirements.txt
```

## Quick Start

### Run Quick Test

Verify installation with a small test case:

```bash
python run_experiments.py --test
```

### Run Single Experiment

```bash
# Experiment 1: Equilibrium structure analysis
python run_experiments.py --exp 1

# Experiment 2: Potential-compatibility
python run_experiments.py --exp 2

# Experiment 3: Learning dynamics
python run_experiments.py --exp 3

# Experiment 4: Coalition stability
python run_experiments.py --exp 4
```

### Run All Experiments

```bash
python run_experiments.py --all
```

### Generate Visualizations

```bash
python run_experiments.py --viz
```

## Experiment Details

### Experiment 1: Equilibrium Structure

**Goal**: Analyze Nash equilibrium existence and multiplicity across social preference parameters.

**Method**: 
- For each (ρ, σ) ∈ [0, 1.2] × [0, 1.2] grid
- Test standard coalition structures (singleton, pairs, half-split, etc.)
- Run best-response dynamics from multiple initial conditions
- Record convergence rate, equilibrium count, potential values

**Output**: `results/experiment_1_equilibrium_structure.json`

**Key findings**:
- Singleton structures: equilibria independent of (ρ, σ)
- Coarser structures: high sensitivity to out-group spite
- Altruism preserves equilibria, spite destabilizes them

---

### Experiment 2: Potential-Compatibility Analysis

**Goal**: Empirically validate the theoretical potential-compatibility bound.

**Method**:
- For each (ρ, σ) combination
- Sample 2000 random coalition deviations
- Check if utility-improving deviations decrease Rosenthal potential
- Calculate "compatibility score" = P(ΔU_C > 0 ⟹ ΔΦ⁰ ≤ 0)

**Output**: `results/experiment_2_potential_compatibility.json`

**Key findings**:
- Perfect compatibility only near (0, 0)
- Compatibility degrades monotonically with ρ + σ
- Theoretical bound is conservative but valid

---

### Experiment 3: Learning Dynamics

**Goal**: Test convergence of logit learning to Nash equilibria under noise.

**Method**:
- Use paired coalition structure with mild preferences (ρ=0.1, σ=0.1)
- Vary temperature τ ∈ [0.01, 1.0]
- Run 10,000 iterations with 2,000 burn-in
- Measure stationary distribution mass on Nash equilibria

**Output**: `results/experiment_3_learning_dynamics.json`

**Key findings**:
- At τ=0.01: 100% convergence to Nash
- At τ=0.5: ~86% time in equilibrium
- At τ=1.0: ~75% time in equilibrium
- Learning noise substantially degrades coordination

---

### Experiment 4: Coalition Stability

**Goal**: Identify which coalition structures are stable (coalition equilibria).

**Method**:
- For each (ρ, σ) combination
- Test standard + 50 random coalition structures
- Find Nash equilibrium via best-response
- Check split stability (no profitable split)
- Check join stability (no profitable merge)

**Output**: `results/experiment_4_coalition_stability.json`

**Key findings**:
- **Nearly selfish regime**: Structures largely irrelevant
- **Altruistic regime**: Grand coalition becomes stable
- **Spiteful regime**: Singleton partition emerges
- **Factional regime**: Intermediate structures stable

---

## Key Concepts

### Coalition Structure

A partition P = {C₁, C₂, ..., Cᵣ} of players into disjoint coalitions.

**Standard structures**:
- **Singleton**: Each player alone, e.g., `{{0}, {1}, {2}, {3}}`
- **Grand coalition**: All players together, e.g., `{{0,1,2,3}}`
- **Pairs**: Players in groups of two, e.g., `{{0,1}, {2,3}}`
- **Half-split**: Two equal groups, e.g., `{{0,1,2}, {3,4,5}}`

### Social Preferences

Each player i evaluates outcomes via:

```
U_i(s; P) = μ_i(s) + ρ·Σ_{j∈C_i\{i}} μ_j(s) - σ·Σ_{m∉C_i} μ_m(s)
```

where:
- **μ_i(s) = -ℓ_i(s)**: Selfish payoff (negative latency)
- **ρ ≥ 0**: In-group altruism (care about coalition members)
- **σ ≥ 0**: Out-group spite (desire to harm outsiders)

### Coalition Equilibrium

A pair (P*, s*) is a **coalition equilibrium** if:

1. **Action stability**: s* is a Nash equilibrium given P*
   - No coalition can improve by changing its joint action
   
2. **Structural stability**: P* is stable against split/join
   - **Split stability**: No coalition wants to split
   - **Join stability**: No pair of coalitions wants to merge

### Social Preference Regimes

The (ρ, σ) plane divides into four regimes:

1. **Nearly selfish**: ρ + σ ≤ bound
   - Coalition structure is strategically inert
   - Equilibria coincide with selfish Nash

2. **Altruistic**: ρ large, σ small
   - Grand coalition is stable
   - Join deviations dominate

3. **Spiteful**: σ large, ρ small
   - Singleton partition is stable
   - Split deviations dominate

4. **Factional**: Both large and comparable
   - Intermediate-sized coalitions stable
   - Multiple blocs emerge

## Code Examples

### Create a Game

```python
from game import CongestionGame, Resource

# Define resources
resources = [
    Resource(id=0, a=1.0, b=0.0),   # Tight
    Resource(id=1, a=0.5, b=2.0),   # Forgiving
    Resource(id=2, a=1.5, b=1.0),   # Medium
]

# Define coalition structure (pairs)
coalition_structure = [{0, 1}, {2, 3}, {4, 5}]

# Create game
game = CongestionGame(
    n_players=6,
    resources=resources,
    coalition_structure=coalition_structure,
    rho=0.3,    # Moderate altruism
    sigma=0.1   # Low spite
)
```

### Find Nash Equilibrium

```python
from equilibrium import EquilibriumSolver

solver = EquilibriumSolver(game)
profile, converged, iterations, trajectory = solver.best_response_dynamics()

print(f"Converged: {converged} in {iterations} iterations")
print(f"Equilibrium: {profile}")
print(f"Social cost: {game.social_cost(profile):.2f}")
```

### Check Stability

```python
from stability import StabilityChecker

checker = StabilityChecker(game)

# Test split stability
split_stable, split_info = checker.test_split_stability(profile, verbose=True)

# Test join stability
join_stable, join_info = checker.test_join_stability(profile, verbose=True)

# Full coalition equilibrium check
is_eq, reason, info = checker.is_coalition_equilibrium(profile, verbose=True)
```

### Run Logit Learning

```python
from equilibrium import LogitLearning

learner = LogitLearning(game, temperature=0.1)
trajectory, potentials = learner.run_dynamics(n_iterations=5000)

# Analyze stationary distribution
stationary = learner.stationary_distribution(n_samples=10000)
print(f"Found {len(stationary)} distinct states")
```

## Advanced Usage

### Custom Parameter Grid

```python
from utils import create_parameter_grid

# Fine-grained grid
custom_grid = create_parameter_grid(
    rho_values=[0, 0.05, 0.1, 0.15, 0.2],
    sigma_values=[0, 0.05, 0.1, 0.15, 0.2]
)

suite = ExperimentSuite(n_players=6)
suite.experiment_1_equilibrium_structure(parameter_grid=custom_grid)
```

### Custom Coalition Structures

```python
from stability import enumerate_coalition_structures

# Enumerate all partitions (warning: exponential!)
all_structures = enumerate_coalition_structures(n_players=5)
print(f"Total partitions: {len(all_structures)}")  # Bell number B(5) = 52

# Or sample random structures
from stability import sample_random_coalition_structures
random_structures = sample_random_coalition_structures(
    n_players=6, 
    n_samples=100, 
    seed=42
)
```

### Pretty Print Game State

```python
from utils import pretty_print_game_state

pretty_print_game_state(game, profile, title="Current Equilibrium")
```

## Output Files

### Results Directory

All experimental results are saved as JSON in `results/`:

- `experiment_1_equilibrium_structure.json`
- `experiment_2_potential_compatibility.json`
- `experiment_3_learning_dynamics.json`
- `experiment_4_coalition_stability.json`

Each result entry contains:
- **Parameters**: ρ, σ, regime
- **Structure**: Coalition structure details
- **Convergence**: Convergence rate, iterations
- **Equilibria**: Count, potential values
- **Stability**: Split/join stability flags
- **Metrics**: Social cost, congestion distribution

### Figures Directory

Visualizations are saved as PNG in `figures/`:

- `equilibria_comparison.png`: Heatmaps of equilibrium counts
- `potential_compatibility.png`: Compatibility score heatmap
- `learning_convergence.png`: Nash mass vs. temperature
- `coalition_robustness_*.png`: Stability pie charts
- `stability_by_regime.png`: Grouped bar chart

## Performance Notes

### Computational Complexity

- **Nash equilibrium search**: O(K^n) exhaustive, O(n·K^|C|) per BR iteration
- **Stability testing**: O(2^|C|) splits per coalition, O(|P|²) joins
- **All partitions**: Bell number B(n) grows super-exponentially
  - B(4) = 15, B(5) = 52, B(6) = 203, B(7) = 877, B(8) = 4140

### Recommended Settings

For quick testing:
- `n_players = 4`
- Standard structures only
- Coarse parameter grid

For publication results:
- `n_players = 6`
- 50-100 random structures
- Fine parameter grid (36+ points)
- Multiple initial conditions (10+)

Expected runtime for full suite (n=6):
- **Experiment 1**: ~10-15 minutes
- **Experiment 2**: ~5-8 minutes
- **Experiment 3**: ~3-5 minutes
- **Experiment 4**: ~30-45 minutes
- **Total**: ~1 hour

## Extending the Code

### Add New Resource Configurations

Edit `utils.py`:

```python
def create_standard_resources(resource_type: str):
    if resource_type == 'my_custom':
        return [
            Resource(id=0, a=2.0, b=0.5),
            Resource(id=1, a=1.0, b=1.0),
            # ...
        ]
```

### Add New Experiment

Edit `experiments.py`:

```python
def experiment_5_my_analysis(self):
    """Your custom experiment."""
    # Implementation here
    pass
```

### Custom Metrics

Edit `utils.py`:

```python
def my_custom_metric(game, profile):
    """Calculate custom metric."""
    # Your logic
    return value
```

## Troubleshooting

### Common Issues

**Issue**: "No module named 'game'"
- **Solution**: Ensure you're in the `new/` directory or adjust Python path

**Issue**: Experiments run very slowly
- **Solution**: Reduce `n_players`, use coarser parameter grid, or limit structures

**Issue**: Visualizations fail
- **Solution**: Ensure experiments have completed and JSON files exist in `results/`

**Issue**: Memory error for large n
- **Solution**: Use sampling instead of exhaustive enumeration

## Citation

If you use this code, please cite:

```bibtex
@article{li2025coalitional,
  title={Coalitional Stability in Congestion Games with Altruism and Spite},
  author={Li, Junxiang and Chen, Jiayi and Xu, Yujie and Qiu, Yiqi},
  journal={},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please contact the authors or open an issue on GitHub.

