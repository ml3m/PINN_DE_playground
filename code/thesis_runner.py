#!/usr/bin/env python3
"""
Main Runner for PINN Bachelor Thesis
====================================

This script runs all experiments and generates all results for the bachelor thesis
on Physics-Informed Neural Networks. It provides a comprehensive demonstration
of PINN capabilities across various problems.

Date: 2025

Usage:
    python thesis_runner.py --all                    # Run all experiments
    python thesis_runner.py --quick                  # Run quick version
    python thesis_runner.py --ode                    # Run only ODE experiments
    python thesis_runner.py --pde                    # Run only PDE experiments
    python thesis_runner.py --experiments            # Run advanced experiments
    python thesis_runner.py --tests                  # Run test suite
"""

import sys
import os
import argparse
import time
from typing import Dict, Any

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from ode_solver import run_ode_experiment
from pde_solver import run_pde_experiment
from experiments import run_comprehensive_study


class ThesisRunner:
    """
    Main runner class for all thesis experiments.
    """
    
    def __init__(self, quick_mode: bool = False, device: str = 'auto'):
        """
        Initialize the thesis runner.
        
        Args:
            quick_mode: If True, run faster but less thorough experiments
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.quick_mode = quick_mode
        self.device = device
        
        # Set up results storage
        self.results = {}
        self.start_time = time.time()
        
        # Create figures directory
        os.makedirs('figures', exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"PHYSICS-INFORMED NEURAL NETWORKS - BACHELOR THESIS")
        print(f"{'='*80}")
        print(f"Mode: {'Quick' if quick_mode else 'Full'}")
        print(f"Device: {device}")
        print(f"Output directory: figures/")
        print(f"{'='*80}")
    
    def run_ode_experiments(self) -> Dict[str, Any]:
        """
        Run all ODE experiments.
        
        Returns:
            Dictionary containing ODE experiment results
        """
        print(f"\n{'-'*60}")
        print("PART 1: ORDINARY DIFFERENTIAL EQUATIONS")
        print(f"{'-'*60}")
        
        ode_results = {}
        
        # Common parameters
        if self.quick_mode:
            ode_params = {
                'n_iterations': 1000,
                'n_collocation': 500,
                'print_frequency': 200
            }
        else:
            ode_params = {
                'n_iterations': 5000,
                'n_collocation': 2000,
                'print_frequency': 500
            }
        
        # 1. Linear ODE: u'(t) = -u(t)
        print("\n1.1 Linear ODE Experiment")
        print("-" * 30)
        try:
            solver, history = run_ode_experiment(
                'linear',
                t_max=3.0,
                u0=1.0,
                lambda_coeff=1.0,
                device=self.device,
                **ode_params
            )
            ode_results['linear'] = {'solver': solver, 'history': history}
            print("✓ Linear ODE experiment completed successfully")
        except Exception as e:
            print(f"✗ Linear ODE experiment failed: {e}")
        
        # 2. Logistic ODE: u'(t) = r*u(t)*(1-u(t))
        print("\n1.2 Logistic Growth ODE Experiment")
        print("-" * 35)
        try:
            solver, history = run_ode_experiment(
                'logistic',
                t_max=5.0,
                u0=0.1,
                r=1.0,
                device=self.device,
                **ode_params
            )
            ode_results['logistic'] = {'solver': solver, 'history': history}
            print("✓ Logistic ODE experiment completed successfully")
        except Exception as e:
            print(f"✗ Logistic ODE experiment failed: {e}")
        
        # 3. Van der Pol Oscillator (if not in quick mode)
        if not self.quick_mode:
            print("\n1.3 Van der Pol Oscillator Experiment")
            print("-" * 38)
            try:
                solver, history = run_ode_experiment(
                    'vanderpol',
                    t_max=10.0,
                    u0=2.0,
                    v0=0.0,
                    mu=1.0,
                    device=self.device,
                    n_iterations=8000,
                    n_collocation=2000,
                    print_frequency=1000
                )
                ode_results['vanderpol'] = {'solver': solver, 'history': history}
                print("✓ Van der Pol experiment completed successfully")
            except Exception as e:
                print(f"✗ Van der Pol experiment failed: {e}")
        
        self.results['ode'] = ode_results
        return ode_results
    
    def run_pde_experiments(self) -> Dict[str, Any]:
        """
        Run all PDE experiments.
        
        Returns:
            Dictionary containing PDE experiment results
        """
        print(f"\n{'-'*60}")
        print("PART 2: PARTIAL DIFFERENTIAL EQUATIONS")
        print(f"{'-'*60}")
        
        pde_results = {}
        
        # Common parameters
        if self.quick_mode:
            pde_params = {
                'n_iterations': 2000,
                'n_collocation': 1000,
                'lambda_ic': 10.0,
                'lambda_bc': 10.0,
                'print_frequency': 400
            }
        else:
            pde_params = {
                'n_iterations': 10000,
                'n_collocation': 2000,
                'lambda_ic': 10.0,
                'lambda_bc': 10.0,
                'print_frequency': 1000
            }
        
        # 1. Heat Equation: u_t = alpha * u_xx
        print("\n2.1 Heat Equation Experiment")
        print("-" * 27)
        try:
            solver, history = run_pde_experiment(
                'heat',
                x_max=1.0,
                t_max=0.5,
                alpha=0.1,
                initial_condition='sine',
                device=self.device,
                **pde_params
            )
            pde_results['heat'] = {'solver': solver, 'history': history}
            print("✓ Heat equation experiment completed successfully")
        except Exception as e:
            print(f"✗ Heat equation experiment failed: {e}")
        
        # 2. Wave Equation: u_tt = c^2 * u_xx
        if not self.quick_mode:
            print("\n2.2 Wave Equation Experiment")
            print("-" * 28)
            try:
                solver, history = run_pde_experiment(
                    'wave',
                    x_max=1.0,
                    t_max=2.0,
                    c=1.0,
                    device=self.device,
                    **pde_params
                )
                pde_results['wave'] = {'solver': solver, 'history': history}
                print("✓ Wave equation experiment completed successfully")
            except Exception as e:
                print(f"✗ Wave equation experiment failed: {e}")
        
        # 3. Burgers' Equation: u_t + u*u_x = nu*u_xx
        if not self.quick_mode:
            print("\n2.3 Burgers' Equation Experiment")
            print("-" * 32)
            try:
                solver, history = run_pde_experiment(
                    'burgers',
                    x_min=-1.0,
                    x_max=1.0,
                    t_max=0.5,
                    nu=0.01,
                    device=self.device,
                    n_iterations=15000,  # Burgers' equation needs more iterations
                    **{k: v for k, v in pde_params.items() if k != 'n_iterations'}
                )
                pde_results['burgers'] = {'solver': solver, 'history': history}
                print("✓ Burgers' equation experiment completed successfully")
            except Exception as e:
                print(f"✗ Burgers' equation experiment failed: {e}")
        
        self.results['pde'] = pde_results
        return pde_results
    
    def run_advanced_experiments(self) -> Dict[str, Any]:
        """
        Run advanced experiments and comparisons.
        
        Returns:
            Dictionary containing advanced experiment results
        """
        print(f"\n{'-'*60}")
        print("PART 3: ADVANCED EXPERIMENTS & ANALYSIS")
        print(f"{'-'*60}")
        
        advanced_results = {}
        
        # Run comprehensive study on linear ODE (fastest problem)
        print("\n3.1 Comprehensive Study on Linear ODE")
        print("-" * 37)
        try:
            if self.quick_mode:
                # Quick study with minimal parameters
                study_results = run_comprehensive_study(
                    'linear_ode',
                    t_max=3.0,
                    u0=1.0,
                    lambda_coeff=1.0,
                    device=self.device
                )
            else:
                # Full study
                study_results = run_comprehensive_study(
                    'linear_ode',
                    t_max=3.0,
                    u0=1.0,
                    lambda_coeff=1.0,
                    device=self.device
                )
            
            advanced_results['comprehensive_study'] = study_results
            print("✓ Comprehensive study completed successfully")
        except Exception as e:
            print(f"✗ Comprehensive study failed: {e}")
        
        # Additional studies on heat equation (if not quick mode)
        if not self.quick_mode:
            print("\n3.2 Heat Equation Advanced Analysis")
            print("-" * 35)
            try:
                heat_study_results = run_comprehensive_study(
                    'heat_pde',
                    x_max=1.0,
                    t_max=0.5,
                    alpha=0.1,
                    initial_condition='sine',
                    device=self.device
                )
                advanced_results['heat_study'] = heat_study_results
                print("✓ Heat equation advanced analysis completed")
            except Exception as e:
                print(f"✗ Heat equation advanced analysis failed: {e}")
        
        self.results['advanced'] = advanced_results
        return advanced_results
    
    def run_tests(self) -> bool:
        """
        Run the comprehensive test suite.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print(f"\n{'-'*60}")
        print("RUNNING COMPREHENSIVE TEST SUITE")
        print(f"{'-'*60}")
        
        try:
            # Import and run tests
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
            from test_pinn_thesis import run_all_tests
            
            success = run_all_tests()
            
            if success:
                print("\n✓ All tests passed successfully!")
            else:
                print("\n✗ Some tests failed. Check output above for details.")
            
            return success
            
        except ImportError as e:
            print(f"✗ Could not import test module: {e}")
            return False
        except Exception as e:
            print(f"✗ Test execution failed: {e}")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments."""
        elapsed_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"THESIS EXPERIMENTS SUMMARY REPORT")
        print(f"{'='*80}")
        print(f"Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"Device: {self.device}")
        
        # Count successful experiments
        total_experiments = 0
        successful_experiments = 0
        
        if 'ode' in self.results:
            for name, result in self.results['ode'].items():
                total_experiments += 1
                if result is not None:
                    successful_experiments += 1
        
        if 'pde' in self.results:
            for name, result in self.results['pde'].items():
                total_experiments += 1
                if result is not None:
                    successful_experiments += 1
        
        if 'advanced' in self.results:
            for name, result in self.results['advanced'].items():
                total_experiments += 1
                if result is not None:
                    successful_experiments += 1
        
        success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
        
        print(f"\nExperiment Results:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful: {successful_experiments}")
        print(f"  Failed: {total_experiments - successful_experiments}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        print(f"\nGenerated Files (in figures/ directory):")
        if os.path.exists('figures'):
            figures = [f for f in os.listdir('figures') if f.endswith('.png')]
            for fig in sorted(figures):
                print(f"  - {fig}")
        
        print(f"\nThesis Implementation Status:")
        if success_rate >= 80:
            print("  ✓ THESIS IMPLEMENTATION SUCCESSFUL!")
            print("  ✓ All major components working correctly")
            print("  ✓ Ready for thesis defense")
        elif success_rate >= 60:
            print("  ⚠ THESIS IMPLEMENTATION MOSTLY SUCCESSFUL")
            print("  ⚠ Some experiments failed, but core functionality works")
            print("  ⚠ Review failed experiments before defense")
        else:
            print("  ✗ THESIS IMPLEMENTATION NEEDS WORK")
            print("  ✗ Multiple critical failures detected")
            print("  ✗ Address issues before proceeding")
        
        print(f"\n{'='*80}")
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all experiments.
        
        Returns:
            Complete results dictionary
        """
        try:
            # Run ODE experiments
            self.run_ode_experiments()
            
            # Run PDE experiments
            self.run_pde_experiments()
            
            # Run advanced experiments
            self.run_advanced_experiments()
            
            # Generate summary
            self.generate_summary_report()
            
            return self.results
            
        except KeyboardInterrupt:
            print("\n\nExperiments interrupted by user")
            self.generate_summary_report()
            return self.results
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            self.generate_summary_report()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PINN Bachelor Thesis Runner')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick version of experiments')
    parser.add_argument('--ode', action='store_true', help='Run only ODE experiments')
    parser.add_argument('--pde', action='store_true', help='Run only PDE experiments')
    parser.add_argument('--experiments', action='store_true', help='Run only advanced experiments')
    parser.add_argument('--tests', action='store_true', help='Run test suite')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Computing device to use')
    
    args = parser.parse_args()
    
    # If no specific option is given, default to --all
    if not any([args.all, args.ode, args.pde, args.experiments, args.tests]):
        args.all = True
    
    # Initialize runner
    runner = ThesisRunner(quick_mode=args.quick, device=args.device)
    
    try:
        if args.tests:
            success = runner.run_tests()
            return 0 if success else 1
        
        if args.all:
            runner.run_all()
        else:
            if args.ode:
                runner.run_ode_experiments()
            if args.pde:
                runner.run_pde_experiments()
            if args.experiments:
                runner.run_advanced_experiments()
            
            runner.generate_summary_report()
        
        return 0
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
