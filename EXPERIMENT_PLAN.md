# Detailed Experimental Plan
# Coalitional Stability in Congestion Games with Altruism and Spite

## Overview

This document outlines the complete experimental plan aligned with the theoretical framework in the paper. The experiments systematically explore how social preferences (in-group altruism ρ and out-group spite σ) affect coalition formation and stability in congestion games.

---

## Phase 1: Environment Setup

### 1.1 Game Parameters

**Players**: n = 6 (default)
- Balances computational tractability with complexity
- Allows diverse coalition structures (Bell number B(6) = 203)

**Resources**: m = 3 with linear latency functions
```
Resource 1: f₁(n) = 1.0n + 0.0  (Tight)
Resource 2: f₂(n) = 0.5n + 2.0  (Forgiving)
Resource 3: f₃(n) = 1.5n + 1.0  (Medium)
```

**Rationale**:
- Heterogeneous resources create strategic trade-offs
- Tight resource punishes large coalitions → reveals spite
- Forgiving resource tolerates large groups → reveals altruism
- Medium resource creates factional incentives

### 1.2 Social Preference Parameter Space

**Grid**: (ρ, σ) ∈ [0, 1.2] × [0, 1.2]

**Sampling points**:
- ρ ∈ {0.0, 0.1, 0.3, 0.5, 0.8, 1.2}
- σ ∈ {0.0, 0.1, 0.3, 0.5, 0.8, 1.2}
- Total: 36 parameter combinations

**Theoretical regimes**:
1. **Nearly selfish**: ρ + σ ≤ α_min/(2n²α_max) ≈ 0.0046
2. **Altruistic**: ρ ≥ 0.5, σ ≤ 0.3ρ
3. **Spiteful**: σ ≥ 0.5, ρ ≤ 0.3σ
4. **Factional**: ρ ≥ 0.5, σ ≥ 0.5

---

## Phase 2: Coalition Structure Design

### 2.1 Standard Structures

**Purpose**: Capture qualitatively different coordination patterns

| Structure | Definition | Example (n=6) | Interpretation |
|-----------|------------|---------------|----------------|
| Singleton | Each player alone | {{0},{1},{2},{3},{4},{5}} | Non-cooperative baseline |
| Grand | All players together | {{0,1,2,3,4,5}} | Full cooperation |
| Pairs | Groups of two | {{0,1},{2,3},{4,5}} | Bilateral partnerships |
| Half-split | Two equal groups | {{0,1,2},{3,4,5}} | Two competing factions |
| Thirds | Three equal groups | {{0,1},{2,3},{4,5}} | Multi-polar competition |
| Asymmetric | One large + small | {{0},{1,2,3,4,5}} | Dominant coalition |

### 2.2 Random Sampling

**Method**: Uniform sampling from all partitions
- Sample size: 50 random structures per experiment
- Ensures coverage of Bell(6) = 203 total partitions
- Use restricted growth strings for efficient generation

**Validation**:
- Ensure partition property (disjoint, complete cover)
- Track size distribution (min, max, mean coalition size)

---

## Phase 3: Nash Equilibrium Computation

### 3.1 Algorithm: Best-Response Dynamics

**Input**: Initial strategy profile s₀ (random or heuristic)

**Iteration**:
```
For each coalition C ∈ P:
    1. Compute current utility U_C(s)
    2. Find best joint response: s'_C = argmax U_C((s'_C, s_{-C}))
    3. If U_C((s'_C, s_{-C})) > U_C(s):
        Update: s ← (s'_C, s_{-C})
        Set improved = True
```

**Termination**:
- Stop if no improvement made (Nash equilibrium found)
- Maximum iterations: 1000

### 3.2 Multiple Initial Conditions

**Purpose**: Detect multiple equilibria and verify uniqueness

**Method**:
- For each (P, ρ, σ) configuration
- Run best-response from 10 random initial profiles
- Deduplicate equilibria by comparing strategy vectors

**Recording**:
- All unique equilibria found
- Rosenthal potential Φ⁰(s*) for each
- Convergence time (iterations)

### 3.3 Potential-Compatibility Verification

**Check**: Does ΔU_C > 0 ⟹ ΔΦ⁰ ≤ 0?

**Method**:
```python
For deviation (C, s'_C):
    delta_U = U_C((s'_C, s_{-C})) - U_C(s)
    delta_Phi = Φ⁰((s'_C, s_{-C})) - Φ⁰(s)
    
    if delta_U > 0 and delta_Phi > 0:
        # Violation of compatibility
        return False
```

**Application**: Used to validate theoretical bounds

---

## Phase 4: Structural Stability Testing

### 4.1 Split Deviation Test

**Definition**: Coalition C splits into C_a ∪ C_b

**Algorithm**:
```
For each coalition C ∈ P:
    For each bipartition (C_a, C_b) of C:
        1. Create new structure P' = (P \ {C}) ∪ {C_a, C_b}
        2. Create new game with P'
        3. Compute utilities:
            U_old = U_C(s; P)
            U_new = U_{C_a}(s; P') + U_{C_b}(s; P')
        4. If U_new > U_old:
            Return (False, profitable_split_info)
    
Return (True, None)
```

**Note**: Strategy profile s remains fixed (immediate utility change)

### 4.2 Join Deviation Test

**Definition**: Coalitions C_a, C_b merge into C_ab = C_a ∪ C_b

**Algorithm**:
```
For each pair (C_a, C_b) ∈ P × P, C_a ≠ C_b:
    1. Create new structure P' = (P \ {C_a, C_b}) ∪ {C_ab}
    2. Create new game with P'
    3. Compute utilities:
        U_old = U_{C_a}(s; P) + U_{C_b}(s; P)
        U_new = U_{C_ab}(s; P')
    4. If U_new > U_old:
        Return (False, profitable_join_info)

Return (True, None)
```

### 4.3 Coalition Equilibrium Definition

A pair (P*, s*) is a **coalition equilibrium** iff:

1. **Action stability**: s* is coalitional Nash given P*
   ```
   ∀C ∈ P*, ∀s'_C: U_C(s*; P*) ≥ U_C((s'_C, s*_{-C}); P*)
   ```

2. **Structural stability**: No profitable split or join
   ```
   Split-stable: ∀splits → ∑U_{split} ≤ U_{original}
   Join-stable:  ∀joins → U_{merged} ≤ ∑U_{separate}
   ```

---

## Phase 5: Logit Learning Dynamics (Optional)

### 5.1 Logit Response Protocol

**Coalition logit probabilities**:
```
P(s'_C | s_{-C}) ∝ exp(U_C((s'_C, s_{-C})) / τ)
```

where τ is temperature parameter (lower = more rational)

### 5.2 Dynamics Algorithm

```
Initialize: s₀ ~ random
For t = 1 to T:
    1. Select coalition C ~ Uniform(P)
    2. Sample s'_C ~ Logit(U_C(·, s_{-C}), τ)
    3. Update: s ← (s'_C, s_{-C})
```

### 5.3 Stationary Distribution Analysis

**Burn-in**: Discard first 2000 iterations

**Sampling**: Collect 10,000 samples

**Metrics**:
- Nash mass = P(state is Nash equilibrium)
- Entropy = -∑ p_i log p_i
- Number of visited states
- Convergence rate vs. temperature

---

## Phase 6: Data Collection

### 6.1 Per-Configuration Metrics

For each (P, ρ, σ, s*):

**Equilibrium metrics**:
- `converged`: Boolean
- `convergence_iterations`: Int
- `n_unique_equilibria`: Int
- `rosenthal_potential`: Float
- `potential_trajectory`: List[Float]

**Stability metrics**:
- `split_stable`: Boolean
- `join_stable`: Boolean
- `is_coalition_equilibrium`: Boolean
- `split_deviation_info`: Dict (if unstable)
- `join_deviation_info`: Dict (if unstable)

**Efficiency metrics**:
- `social_cost`: ∑ᵢ ℓᵢ(s*)
- `max_latency`: max_i ℓᵢ(s*)
- `congestion_distribution`: [n₁, n₂, n₃]
- `resource_usage_entropy`: -∑ (nₖ/n) log(nₖ/n)

**Coalition metrics**:
- `n_coalitions`: |P|
- `coalition_sizes`: [|C₁|, |C₂|, ...]
- `max_coalition_size`: max |C|
- `mean_coalition_size`: mean |C|
- `gini_coefficient`: Coalition size inequality

### 6.2 Cross-Configuration Analysis

**Regime comparison**:
- Group by regime (nearly_selfish, altruistic, spiteful, factional)
- Compare stability rates, social costs, equilibrium counts

**Structure comparison**:
- For fixed (ρ, σ), compare across structures
- Identify which structures are stable

**Parameter sensitivity**:
- For fixed P, vary (ρ, σ)
- Track phase transitions in stability

---

## Phase 7: Specific Experimental Protocols

### Experiment 1: Equilibrium Structure Across Parameters

**Goal**: Map equilibrium landscape in (ρ, σ) space

**Procedure**:
```
For each (ρ, σ) in parameter_grid:
    For each structure in standard_structures:
        For trial in 1..10:
            s₀ = random_initial()
            s*, converged = best_response_dynamics(s₀)
            if converged:
                record(s*, Φ⁰(s*), metrics)
        
        deduplicate_equilibria()
        save_results()
```

**Expected output**:
- Heatmap: equilibrium count vs. (ρ, σ) for each structure
- Finding: Singleton structure → constant count
- Finding: Coarse structures → declining count with σ

**Runtime**: ~10-15 minutes (n=6)

---

### Experiment 2: Potential-Compatibility Empirical Validation

**Goal**: Verify theoretical bound ρ + σ ≤ α_min/(2n²α_max)

**Procedure**:
```
For each (ρ, σ) in fine_grid:
    compatible_count = 0
    improving_count = 0
    
    For sample in 1..2000:
        s = random_profile()
        C = random_coalition()
        s'_C = random_deviation()
        
        ΔU = U_C((s'_C, s_{-C})) - U_C(s)
        ΔΦ = Φ⁰((s'_C, s_{-C})) - Φ⁰(s)
        
        if ΔU > 0:
            improving_count += 1
            if ΔΦ ≤ 0:
                compatible_count += 1
    
    score = compatible_count / improving_count
    save_result(ρ, σ, score)
```

**Expected output**:
- Heatmap: compatibility score vs. (ρ, σ)
- Finding: Score ≈ 1.0 near (0,0)
- Finding: Degradation follows ρ + σ
- Overlay: Theoretical bound line

**Runtime**: ~5-8 minutes

---

### Experiment 3: Learning Convergence Under Noise

**Goal**: Quantify how learning noise affects equilibrium selection

**Procedure**:
```
structure = pairs
ρ, σ = 0.1, 0.1

For τ in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    # Find Nash equilibria first
    nash_set = find_all_nash_equilibria()
    
    # Run logit learning
    learner = LogitLearning(τ)
    stationary = learner.stationary_distribution(
        n_samples=10000,
        burn_in=2000
    )
    
    # Calculate Nash mass
    nash_mass = sum(stationary[s] for s in nash_set)
    
    save_result(τ, nash_mass, len(nash_set))
```

**Expected output**:
- Line plot: Nash mass vs. τ
- Finding: Perfect convergence at τ→0
- Finding: Linear degradation with τ
- Bar chart: Number of stable states vs. τ

**Runtime**: ~3-5 minutes

---

### Experiment 4: Coalition Stability Robustness

**Goal**: Identify stable coalition structures for each (ρ, σ)

**Procedure**:
```
For each (ρ, σ) in parameter_grid:
    For each structure in (standard + 50 random):
        # Find Nash equilibrium
        game = CongestionGame(P=structure, ρ=ρ, σ=σ)
        s* = find_nash_equilibrium(game)
        
        if not converged:
            continue
        
        # Test stability
        split_stable = test_split_stability(game, s*)
        join_stable = test_join_stability(game, s*)
        is_eq = split_stable and join_stable
        
        save_result({
            'rho': ρ, 'sigma': σ,
            'structure': structure,
            'split_stable': split_stable,
            'join_stable': join_stable,
            'is_coalition_eq': is_eq,
            'social_cost': social_cost(s*)
        })
```

**Expected output**:
- By regime:
  - **Nearly selfish**: All structures ~equally stable
  - **Altruistic (ρ=0.8, σ=0.1)**: Grand coalition stable
  - **Spiteful (ρ=0.1, σ=0.8)**: Singleton stable
  - **Factional (ρ=0.6, σ=0.6)**: 2-3 coalition structures stable

- Pie chart: Stability distribution for each regime
- Bar chart: Stability rate by regime

**Runtime**: ~30-45 minutes

---

## Phase 8: Visualization Plan

### 8.1 Core Visualizations

**Figure 1: Equilibrium Count Heatmaps**
- 3 panels: singleton, pairs, half-split structures
- X-axis: ρ, Y-axis: σ
- Color: number of equilibria
- Annotations: actual counts in cells

**Figure 2: Potential-Compatibility Landscape**
- Heatmap: compatibility score
- Color scale: 0 (red) to 1 (green)
- Overlay: theoretical bound line (ρ + σ = threshold)
- Gradient shows degradation from origin

**Figure 3: Learning Convergence**
- Left panel: Nash mass vs. temperature
- Right panel: # stable states vs. temperature
- Annotations: exact values at key points

**Figure 4: Coalition Robustness**
- Pie chart: stable/unstable structures for (ρ=0.1, σ=0.1)
- Categories: coalition_eq, split_stable_only, join_stable_only, unstable
- Explode coalition_eq slice

**Figure 5: Stability by Regime**
- Grouped bar chart
- X-axis: regimes (nearly_selfish, altruistic, spiteful, factional)
- Y-axis: stability rate
- 3 bars per regime: split_stable, join_stable, coalition_eq

### 8.2 Supplementary Visualizations

**Figure S1**: Convergence time vs. (ρ, σ)
**Figure S2**: Social cost heatmaps by structure
**Figure S3**: Potential trajectories for sample runs
**Figure S4**: Resource usage distribution by regime
**Figure S5**: Coalition size distribution by regime

---

## Phase 9: Statistical Analysis

### 9.1 Hypothesis Tests

**H1**: Altruism preserves equilibria
- Compare equilibrium counts: ρ=0 vs. ρ=0.8 (σ=0)
- Expected: No significant difference

**H2**: Spite destabilizes equilibria
- Compare equilibrium counts: σ=0 vs. σ=0.8 (ρ=0)
- Expected: Significant decrease

**H3**: Grand coalition stable in altruistic regime
- Test: P={{N}} is coalition_eq when ρ ≥ 0.5, σ ≤ 0.15
- Expected: >80% stable

**H4**: Singleton stable in spiteful regime
- Test: P={{i} for all i} is coalition_eq when σ ≥ 0.5, ρ ≤ 0.15
- Expected: >80% stable

### 9.2 Regression Analysis

**Model**: stability ~ ρ + σ + ρ·σ + structure_features

**Features**:
- `n_coalitions`: number of coalitions
- `max_size`: largest coalition size
- `size_variance`: inequality in coalition sizes

**Goal**: Quantify relative impact of parameters on stability

---

## Phase 10: Computational Considerations

### 10.1 Complexity Analysis

| Component | Complexity | Notes |
|-----------|------------|-------|
| Nash search (exhaustive) | O(K^n) | Infeasible for n>6 |
| Best-response iteration | O(n · K^{max\|C\|}) | Per iteration |
| Split stability | O(2^{\|C\|} · \|P\|) | Exponential in coalition size |
| Join stability | O(\|P\|²) | Polynomial |
| All partitions | O(B(n)) | Bell number |

### 10.2 Optimization Strategies

**Parallelization**:
- Parameter combinations are independent → parallelize over (ρ, σ)
- Multiple initial conditions → parallelize trials
- Use joblib or multiprocessing

**Caching**:
- Store computed utilities
- Reuse congestion calculations
- Cache potential values

**Pruning**:
- Early stopping if no improvement in 50 iterations
- Skip stability test if Nash not found
- Sample structures instead of exhaustive enumeration for n>6

**Approximation** (for large n):
- Use local search instead of best-response
- Sample deviations instead of exhaustive
- Neural network to predict equilibria

### 10.3 Expected Runtime

For n=6, m=3:

| Experiment | Configs | Time per Config | Total Time |
|------------|---------|-----------------|------------|
| Exp 1 | 36 × 6 × 10 | 2-3s | 10-15 min |
| Exp 2 | 36 | 5-10s | 5-8 min |
| Exp 3 | 6 | 30-60s | 3-5 min |
| Exp 4 | 36 × 50 | 1-2s | 30-45 min |
| **Total** | | | **~1 hour** |

With parallelization (4 cores): **~20 minutes**

---

## Phase 11: Validation and Sanity Checks

### 11.1 Theoretical Consistency

**Check 1**: Selfish case recovers classical results
- At ρ=σ=0, equilibria should match Rosenthal's construction
- Potential should be exact for deviations

**Check 2**: Potential-compatibility holds in bound
- For ρ+σ ≤ threshold, compatibility score should be >0.95

**Check 3**: Symmetry
- For symmetric structures, players in same coalition should behave identically

**Check 4**: Monotonicity
- Increasing ρ should not decrease join stability
- Increasing σ should not increase join stability

### 11.2 Numerical Stability

**Check 5**: Convergence consistency
- Running from same initial profile should reach same equilibrium

**Check 6**: Floating-point precision
- Use tolerance 1e-9 for equality checks
- Verify potential calculations with symbolic computation

**Check 7**: Boundary cases
- All players on one resource
- All resources empty except one
- Maximum congestion scenarios

---

## Deliverables

### Code Deliverables
1. `game.py` - Core game implementation ✓
2. `equilibrium.py` - Nash computation ✓
3. `stability.py` - Coalition stability ✓
4. `utils.py` - Utilities and metrics ✓
5. `experiments.py` - Experiment framework ✓
6. `visualization.py` - Plotting tools ✓
7. `run_experiments.py` - Main script ✓
8. `test_core.py` - Unit tests ✓
9. `example_usage.py` - Usage examples ✓

### Documentation Deliverables
1. `README.md` - Setup and usage guide ✓
2. `EXPERIMENT_PLAN.md` - This document ✓
3. `requirements.txt` - Dependencies ✓

### Data Deliverables
1. `results/*.json` - Raw experimental results
2. `figures/*.png` - Visualizations
3. `summary_statistics.json` - Aggregate metrics

### Paper Deliverables
1. Section 5 figures (equilibria, compatibility, learning, stability)
2. Supplementary tables (detailed results)
3. Appendix: Extended data and robustness checks

---

## Timeline

**Week 1**: Code development and testing
- Days 1-3: Core implementation
- Days 4-5: Testing and debugging
- Days 6-7: Example runs and validation

**Week 2**: Experiments and analysis
- Days 1-2: Experiments 1-2
- Days 3-4: Experiments 3-4
- Days 5: Extended analysis
- Days 6-7: Visualization and writeup

**Total**: 2 weeks from code to results

---

## Appendix: Extended Experiments (Optional)

### A.1 Heterogeneous Social Preferences

Allow different (ρᵢ, σᵢ) per player instead of uniform

### A.2 Non-linear Latency Functions

Test polynomial f_k(n) = a_k n² + b_k n + c_k

### A.3 Incomplete Information

Players uncertain about others' preferences

### A.4 Dynamic Coalition Formation

Allow sequential coalition updates during play

### A.5 Network Structure

Constrain coalitions to form on a graph

### A.6 Asymmetric Players

Different player types or weights

---

## References

See `main.tex` for complete bibliography.

Key theoretical results:
- Rosenthal (1973): Potential games
- Monderer & Shapley (1996): Exact potentials
- Chen et al. (2008): Altruism in routing
- Bilò (2013): Generalized altruism
- Fotakis & Spirakis (2006): Coalitional congestion games

