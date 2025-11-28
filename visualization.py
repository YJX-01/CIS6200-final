"""
Visualization tools for coalitional congestion game experiments.

Creates plots for:
- Phase diagrams (rho, sigma)
- Equilibrium counts
- Potential-compatibility heatmaps
- Learning convergence
- Stability landscapes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


class ExperimentVisualizer:
    """Visualize experiment results."""
    
    def __init__(self, results_dir: str = 'results', output_dir: str = 'figures'):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing result JSON files
            output_dir: Directory to save figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, filename: str) -> Optional[List[Dict]]:
        """Load results from JSON file."""
        path = self.results_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def plot_archetype_stability_phases(
        self,
        results_file: str = 'experiment_1_archetype_stability.json'
    ):
        """
        Plot Archetype Stability Phases: Three heatmaps showing where each
        archetype (Grand, Singleton, Factions) is stable.
        
        Validates the four regime hypothesis:
        - Grand stable in high-ρ low-σ (altruistic)
        - Singleton stable in low-ρ high-σ (spiteful)
        - Factions stable in high-ρ high-σ (factional)
        
        Args:
            results_file: Results file to load
        """
        results = self.load_results(results_file)
        
        if results is None:
            print(f"⚠ Cannot plot: {results_file} not found. Run --exp 1 first.")
            return
        
        # Extract parameter grid
        rho_values = sorted(set(r['rho'] for r in results))
        sigma_values = sorted(set(r['sigma'] for r in results))
        
        # Create matrices for each archetype
        archetypes = ['grand', 'singleton', 'factions']
        matrices = {arch: np.zeros((len(sigma_values), len(rho_values))) 
                   for arch in archetypes}
        
        for r in results:
            if not r['converged']:
                continue
            i = sigma_values.index(r['sigma'])
            j = rho_values.index(r['rho'])
            arch = r['archetype']
            matrices[arch][i, j] = 1 if r['is_stable'] else 0
        
        # Create three-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        titles = {
            'grand': 'Grand Coalition',
            'singleton': 'Singleton (Atomization)',
            'factions': 'Two-Faction Split'
        }
        
        for ax, arch in zip(axes, archetypes):
            im = ax.imshow(matrices[arch], cmap='RdYlGn', aspect='auto', origin='lower',
                          vmin=0, vmax=1)
            
            ax.set_xticks(range(0, len(rho_values), max(1, len(rho_values)//5)))
            ax.set_yticks(range(0, len(sigma_values), max(1, len(sigma_values)//5)))
            ax.set_xticklabels([f'{rho_values[i]:.1f}' for i in range(0, len(rho_values), max(1, len(rho_values)//5))])
            ax.set_yticklabels([f'{sigma_values[i]:.1f}' for i in range(0, len(sigma_values), max(1, len(sigma_values)//5))])
            
            ax.set_xlabel('ρ (Altruism)', fontsize=13)
            ax.set_ylabel('σ (Spite)', fontsize=13)
            ax.set_title(titles[arch], fontsize=14, weight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, aspect=40)
        cbar.set_label('Stable (Green) / Unstable (Red)', fontsize=12)
        
        fig.suptitle('Archetype Stability Phase Diagrams', fontsize=16, weight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'archetype_stability_phases.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    def plot_splitting_mechanism(
        self,
        results_file: str = 'experiment_2_splitting_mechanism.json'
    ):
        """
        Plot Splitting Mechanism: How σ induces grand coalition collapse.
        
        Shows:
        - Split incentive Δ V vs. σ
        - Transition from stable to unstable
        - Demonstrates that splitting is motivated by attack capability,
          not efficiency
        
        Args:
            results_file: Results file to load
        """
        results = self.load_results(results_file)
        
        if results is None:
            print(f"⚠ Cannot plot: {results_file} not found. Run --exp 2 first.")
            return
        
        # Extract data
        sigma_values = [r['sigma'] for r in results]
        split_incentives = [r['split_incentive'] for r in results]
        is_stable = [r['is_split_stable'] for r in results]
        social_costs = [r['social_cost_grand'] for r in results]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Split incentive
        colors = ['green' if stable else 'red' for stable in is_stable]
        ax1.plot(sigma_values, split_incentives, 'o-', linewidth=2, markersize=6)
        ax1.scatter(sigma_values, split_incentives, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Mark transition point
        for i in range(1, len(is_stable)):
            if is_stable[i-1] and not is_stable[i]:
                ax1.axvline(x=sigma_values[i], color='orange', linestyle='--', linewidth=2, 
                           label=f'Instability Threshold (σ≈{sigma_values[i]:.2f})')
                break
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax1.set_xlabel('σ (Out-Group Spite)', fontsize=14)
        ax1.set_ylabel('Split Incentive ΔV', fontsize=14)
        ax1.set_title('Mechanism of Grand Coalition Collapse', fontsize=16, weight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add annotations
        ax1.text(0.02, 0.98, 'Stable Zone\n(ΔV ≤ 0)', transform=ax1.transAxes,
                fontsize=11, color='green', weight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.text(0.98, 0.02, 'Unstable Zone\n(ΔV > 0)', transform=ax1.transAxes,
                fontsize=11, color='red', weight='bold', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Social cost (demonstrates it's not about efficiency)
        ax2.plot(sigma_values, social_costs, 's-', linewidth=2, markersize=6, color='purple')
        ax2.set_xlabel('σ (Out-Group Spite)', fontsize=14)
        ax2.set_ylabel('Social Cost (Grand Coalition)', fontsize=14)
        ax2.set_title('Splitting is NOT About Efficiency Loss', fontsize=16, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'splitting_mechanism.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    def plot_price_of_stability(
        self,
        results_file: str = 'experiment_3_price_of_stability.json'
    ):
        """
        Plot Price of Stability: Efficiency analysis across parameter space.
        
        Shows:
        - Which archetype is most stable in each region
        - Cost ratio vs. selfish baseline
        - Demonstrates that altruistic structures improve efficiency,
          while spiteful structures catastrophically worsen it
        
        Args:
            results_file: Results file to load
        """
        results = self.load_results(results_file)
        
        if results is None:
            print(f"⚠ Cannot plot: {results_file} not found. Run --exp 3 first.")
            return
        
        # Extract parameter grid
        rho_values = sorted(set(r['rho'] for r in results))
        sigma_values = sorted(set(r['sigma'] for r in results))
        
        # Create matrices
        cost_ratio_matrix = np.zeros((len(sigma_values), len(rho_values)))
        archetype_matrix = np.zeros((len(sigma_values), len(rho_values)))
        archetype_map = {'grand': 0, 'singleton': 1, 'factions': 2}
        
        for r in results:
            i = sigma_values.index(r['sigma'])
            j = rho_values.index(r['rho'])
            cost_ratio_matrix[i, j] = r['cost_ratio']
            archetype_matrix[i, j] = archetype_map.get(r['best_archetype'], -1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Cost ratio heatmap
        im1 = ax1.imshow(cost_ratio_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                        vmin=0.5, vmax=2.0)
        
        ax1.set_xticks(range(0, len(rho_values), max(1, len(rho_values)//5)))
        ax1.set_yticks(range(0, len(sigma_values), max(1, len(sigma_values)//5)))
        ax1.set_xticklabels([f'{rho_values[i]:.1f}' for i in range(0, len(rho_values), max(1, len(rho_values)//5))])
        ax1.set_yticklabels([f'{sigma_values[i]:.1f}' for i in range(0, len(sigma_values), max(1, len(sigma_values)//5))])
        
        ax1.set_xlabel('ρ (Altruism)', fontsize=14)
        ax1.set_ylabel('σ (Spite)', fontsize=14)
        ax1.set_title('Cost Ratio: Stable Structure / Selfish Baseline', fontsize=16, weight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Cost Ratio (<1: Better, >1: Worse)', fontsize=12)
        
        # Add 1.0 contour line
        contour = ax1.contour(cost_ratio_matrix, levels=[1.0], colors='black', 
                             linewidths=2, origin='lower')
        ax1.clabel(contour, inline=True, fontsize=10)
        
        # Plot 2: Dominant archetype
        cmap_arch = plt.cm.get_cmap('Set3', 3)
        im2 = ax2.imshow(archetype_matrix, cmap=cmap_arch, aspect='auto', origin='lower',
                        vmin=-0.5, vmax=2.5)
        
        ax2.set_xticks(range(0, len(rho_values), max(1, len(rho_values)//5)))
        ax2.set_yticks(range(0, len(sigma_values), max(1, len(sigma_values)//5)))
        ax2.set_xticklabels([f'{rho_values[i]:.1f}' for i in range(0, len(rho_values), max(1, len(rho_values)//5))])
        ax2.set_yticklabels([f'{sigma_values[i]:.1f}' for i in range(0, len(sigma_values), max(1, len(sigma_values)//5))])
        
        ax2.set_xlabel('ρ (Altruism)', fontsize=14)
        ax2.set_ylabel('σ (Spite)', fontsize=14)
        ax2.set_title('Most Stable Archetype', fontsize=16, weight='bold')
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap_arch(0), label='Grand Coalition'),
            Patch(facecolor=cmap_arch(1), label='Singleton'),
            Patch(facecolor=cmap_arch(2), label='Factions')
        ]
        ax2.legend(handles=legend_elements, fontsize=11, loc='upper left')
        
        plt.tight_layout()
        output_path = self.output_dir / 'price_of_stability.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    def plot_structure_fragility(
        self,
        results_file: str = 'experiment_4_structure_fragility.json'
    ):
        """
        Plot Structure Fragility: Why only extreme structures are stable.
        
        Compares:
        - Archetype structures (Grand, Singleton, Factions) - stable
        - Random asymmetric structures - almost all unstable
        
        Shows failure breakdown: split/join/both/stable
        
        Args:
            results_file: Results file to load
        """
        results = self.load_results(results_file)
        
        if results is None:
            print(f"⚠ Cannot plot: {results_file} not found. Run --exp 4 first.")
            return
        
        # Separate archetypes and random structures
        arch_results = [r for r in results if r['structure_type'] == 'archetype']
        random_results = [r for r in results if r['structure_type'] == 'random' and r['converged']]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Stability rate comparison
        categories = ['Archetypes\n(n=3)', 'Random\nStructures\n(n=100)']
        
        if arch_results:
            arch_stable_rate = sum(r['is_stable'] for r in arch_results) / len(arch_results)
        else:
            arch_stable_rate = 0
        
        if random_results:
            random_stable_rate = sum(r['is_stable'] for r in random_results) / len(random_results)
        else:
            random_stable_rate = 0
        
        rates = [arch_stable_rate, random_stable_rate]
        colors = ['green', 'red']
        
        bars = ax1.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1%}',
                    ha='center', va='bottom', fontsize=14, weight='bold')
        
        ax1.set_ylabel('Stability Rate', fontsize=14)
        ax1.set_title('Fragility of Generic Coalition Structures', fontsize=16, weight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add significance annotation
        ax1.text(0.5, 0.5, f'Only extreme/symmetric\nstructures are stable!\n\n' + 
                f'Random structures:\n{random_stable_rate:.1%} stable',
                transform=ax1.transAxes, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Failure mode breakdown
        failure_counts = {'stable': 0, 'split': 0, 'join': 0, 'both': 0}
        
        for r in random_results:
            failure_type = r.get('failure_type', 'unknown')
            if failure_type in failure_counts:
                failure_counts[failure_type] += 1
        
        # Pie chart
        labels = []
        sizes = []
        colors_pie = []
        
        if failure_counts['stable'] > 0:
            labels.append(f"Stable\n({failure_counts['stable']})")
            sizes.append(failure_counts['stable'])
            colors_pie.append('green')
        
        if failure_counts['split'] > 0:
            labels.append(f"Split Unstable\n({failure_counts['split']})")
            sizes.append(failure_counts['split'])
            colors_pie.append('orange')
        
        if failure_counts['join'] > 0:
            labels.append(f"Join Unstable\n({failure_counts['join']})")
            sizes.append(failure_counts['join'])
            colors_pie.append('blue')
        
        if failure_counts['both'] > 0:
            labels.append(f"Both Unstable\n({failure_counts['both']})")
            sizes.append(failure_counts['both'])
            colors_pie.append('red')
        
        if sizes:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                                autopct='%1.1f%%', startangle=90,
                                                textprops={'fontsize': 11})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        ax2.set_title('Failure Mode Breakdown\n(Random Structures)', fontsize=16, weight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'structure_fragility.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    def create_all_plots(self):
        """Create all visualizations for the four experiments."""
        print("\n" + "="*70)
        print("Creating Visualizations (Static Stability Analysis)")
        print("="*70 + "\n")
        
        # Check which data files exist
        data_files = {
            'Experiment 1 (Archetype Phases)': 'experiment_1_archetype_stability.json',
            'Experiment 2 (Splitting Mechanism)': 'experiment_2_splitting_mechanism.json',
            'Experiment 3 (Price of Stability)': 'experiment_3_price_of_stability.json',
            'Experiment 4 (Structure Fragility)': 'experiment_4_structure_fragility.json',
        }
        
        available = []
        missing = []
        
        for name, filename in data_files.items():
            if (self.results_dir / filename).exists():
                available.append((name, filename))
            else:
                missing.append((name, filename))
        
        if missing:
            print("⚠ Missing data files:")
            for name, filename in missing:
                exp_num = filename.split('_')[1]
                print(f"  - {name}: Run 'python run_experiments.py --exp {exp_num}'")
            print()
        
        if not available:
            print("✗ No experiment data found. Run experiments first:")
            print("  python run_experiments.py --all")
            return
        
        print(f"Found {len(available)} data file(s). Generating plots...\n")
        
        success_count = 0
        
        # Plot 1: Archetype Stability Phases
        try:
            self.plot_archetype_stability_phases()
            print("✓ Archetype Stability Phases (3-panel heatmaps)")
            success_count += 1
        except Exception as e:
            print(f"✗ Archetype Stability Phases: {e}")
        
        # Plot 2: Splitting Mechanism
        try:
            self.plot_splitting_mechanism()
            print("✓ Splitting Mechanism (split incentive vs σ)")
            success_count += 1
        except Exception as e:
            print(f"✗ Splitting Mechanism: {e}")
        
        # Plot 3: Price of Stability
        try:
            self.plot_price_of_stability()
            print("✓ Price of Stability (cost ratio + dominant archetype)")
            success_count += 1
        except Exception as e:
            print(f"✗ Price of Stability: {e}")
        
        # Plot 4: Structure Fragility
        try:
            self.plot_structure_fragility()
            print("✓ Structure Fragility (archetype vs random stability)")
            success_count += 1
        except Exception as e:
            print(f"✗ Structure Fragility: {e}")
        
        print(f"\n{'='*70}")
        print(f"Successfully generated {success_count}/{len(data_files)} visualizations!")
        print(f"Output directory: {self.output_dir}/")
        print(f"{'='*70}\n")
        
        if success_count == len(data_files):
            print("✅ All visualizations complete! Files generated:")
            print("  - archetype_stability_phases.png")
            print("  - splitting_mechanism.png")
            print("  - price_of_stability.png")
            print("  - structure_fragility.png")
            print()


if __name__ == '__main__':
    viz = ExperimentVisualizer()
    viz.create_all_plots()

