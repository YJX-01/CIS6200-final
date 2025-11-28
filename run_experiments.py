#!/usr/bin/env python
"""
Main script to run all experiments for coalitional congestion games.

Usage:
    python run_experiments.py --all              # Run all experiments
    python run_experiments.py --test             # Run quick test
    python run_experiments.py --exp 1 2          # Run specific experiments
    python run_experiments.py --viz              # Generate visualizations
"""

import argparse
import sys
from experiments import ExperimentSuite, quick_test
from visualization import ExperimentVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Run coalitional congestion game experiments'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all experiments'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run quick test'
    )
    
    parser.add_argument(
        '--exp', 
        nargs='+', 
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific experiments (1-4)'
    )
    
    parser.add_argument(
        '--viz', 
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--n-players', 
        type=int, 
        default=6,
        help='Number of players (default: 6)'
    )
    
    parser.add_argument(
        '--resources', 
        type=str, 
        default='heterogeneous',
        choices=['heterogeneous', 'homogeneous', 'random'],
        help='Resource type (default: heterogeneous)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.test:
        print("Running quick test...")
        quick_test()
        return
    
    # Visualization mode
    if args.viz:
        print("Generating visualizations...")
        viz = ExperimentVisualizer(
            results_dir=args.output_dir,
            output_dir='figures'
        )
        viz.create_all_plots()
        return
    
    # No action specified
    if not args.all and not args.exp:
        parser.print_help()
        print("\nError: Must specify --all, --exp, --test, or --viz")
        sys.exit(1)
    
    # Initialize experiment suite
    suite = ExperimentSuite(
        n_players=args.n_players,
        resource_type=args.resources,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    # Run all experiments
    if args.all:
        suite.run_all_experiments()
        return
    
    # Run specific experiments
    if args.exp:
        for exp_num in sorted(args.exp):
            if exp_num == 1:
                print("\n[Experiment 1] Archetype Stability Phases")
                suite.experiment_1_archetype_stability_phases()
            elif exp_num == 2:
                print("\n[Experiment 2] Splitting Mechanism (Spite Incentive)")
                suite.experiment_2_splitting_mechanism()
            elif exp_num == 3:
                print("\n[Experiment 3] Price of Stability (Efficiency Analysis)")
                suite.experiment_3_price_of_stability()
            elif exp_num == 4:
                print("\n[Experiment 4] Structure Fragility (Generic Instability)")
                suite.experiment_4_fragility_of_generic_structures()
        
        print("\nExperiments completed!")


if __name__ == '__main__':
    main()

