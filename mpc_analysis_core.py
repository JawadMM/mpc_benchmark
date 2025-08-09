"""
Core MPC Analysis Class
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from building_mpc import BuildingEnergyMPC
from data_manager import DataManager
from chart_generator import ChartGenerator
from statistical_analyzer import StatisticalAnalyzer
from report_generator import ReportGenerator

class ComprehensiveMPCAnalysis:
    """
    Comprehensive MPC analysis across all building-weather combinations
    """
    
    def __init__(self, output_dir='mpc_results'):
        self.output_dir = output_dir
        self.results_summary = {}
        self.comparison_summary = {}
        
        # Initialize components
        self.data_manager = DataManager()
        self.chart_generator = ChartGenerator(output_dir)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator(output_dir)
        
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create organized folder structure for results"""
        base_dirs = [
            self.output_dir,
            f'{self.output_dir}/charts',
            f'{self.output_dir}/data',
            f'{self.output_dir}/analysis'
        ]
        
        # Create building-weather combination folders
        buildings = ['Building_1', 'Building_2', 'Building_3']
        weather_locations = ['Abha', 'Jeddah', 'Riyadh']
        
        for building in buildings:
            for weather in weather_locations:
                combo_dir = f'{self.output_dir}/charts/{building}_{weather}'
                base_dirs.append(combo_dir)
                
        for directory in base_dirs:
            os.makedirs(directory, exist_ok=True)
            
        print(f"Created output directory structure in: {self.output_dir}")

    def load_and_prepare_data(self):
        """Load and prepare all CSV data"""
        return self.data_manager.load_and_prepare_data()

    def run_single_simulation(self, building_data, weather_data, building_name, weather_name):
        """Run MPC simulation for single building-weather combination"""
        print(f"\n{'='*70}")
        print(f"SIMULATING {building_name.upper()} with {weather_name.upper()} WEATHER")
        print(f"{'='*70}")
        
        # Initialize MPC controller with location-specific parameters
        location_params = self.get_location_parameters(weather_name)
        
        mpc_controller = BuildingEnergyMPC(
            prediction_horizon=24,
            control_horizon=8,
            sampling_time=1.0,
            comfort_temp_range=(21.0, 25.0),
            comfort_humidity_range=(40.0, 65.0),
            electricity_price=location_params['electricity_price'],
            penalty_comfort=200.0,
            penalty_control_effort=0.5
        )
        
        # Define simulation periods (different seasons)
        simulation_periods = {
            'Winter': (24*30, 24*14),      # 2 weeks in February
            'Spring': (24*90, 24*14),      # 2 weeks in April  
            'Summer': (24*150, 24*14),     # 2 weeks in June
            'Autumn': (24*240, 24*14)      # 2 weeks in September
        }
        
        season_results = {}
        season_metrics = {}
        
        for season_name, (start_time, duration) in simulation_periods.items():
            print(f"\n--- {season_name} Season Analysis ---")
            
            # Check data availability
            if start_time + duration > min(len(building_data), len(weather_data)):
                print(f"Skipping {season_name} - insufficient data")
                continue
            
            try:
                # Run MPC simulation
                results = mpc_controller.simulate_mpc_control(
                    building_data=building_data,
                    weather_data=weather_data,
                    start_time=start_time,
                    simulation_length=duration
                )
                
                # Analyze results
                metrics = mpc_controller.analyze_results(results)
                
                # Create baseline comparison
                baseline_results = self.create_baseline_comparison(
                    building_data, weather_data, start_time, duration
                )
                
                # Calculate performance comparison
                comparison = self.compare_mpc_vs_baseline(results, baseline_results)
                
                season_results[season_name] = {
                    'mpc': results,
                    'baseline': baseline_results,
                    'comparison': comparison
                }
                season_metrics[season_name] = metrics
                
                print(f"  Energy Cost: ${metrics['total_energy_cost']:.2f}")
                print(f"  Comfort Violations: {metrics['violation_rate']:.1f}%")
                print(f"  vs Baseline Savings: {comparison['cost_reduction_pct']:.1f}%")
                
            except Exception as e:
                print(f"Error in {season_name} simulation: {str(e)}")
                continue
        
        # Calculate annual summary
        annual_summary = self.calculate_annual_summary(season_metrics, season_results)
        
        # Save results
        combo_key = f"{building_name}_{weather_name}"
        self.results_summary[combo_key] = {
            'building': building_name,
            'weather': weather_name,
            'seasons': season_metrics,
            'annual': annual_summary,
            'detailed_results': season_results
        }
        
        # Generate and save charts
        self.chart_generator.create_comprehensive_charts(combo_key, season_results, annual_summary)
        
        return annual_summary

    def get_location_parameters(self, weather_name):
        """Get location-specific parameters"""
        location_params = {
            'Abha': {'electricity_price': 0.10},      # Mountain region, cooler
            'Jeddah': {'electricity_price': 0.12},    # Coastal, humid
            'Riyadh': {'electricity_price': 0.11}     # Desert, hot and dry
        }
        return location_params.get(weather_name, {'electricity_price': 0.12})

    def create_baseline_comparison(self, building_data, weather_data, start_time, duration):
        """Create baseline thermostat control for comparison"""
        baseline_results = {
            'time': [], 'indoor_temperature': [], 'indoor_humidity': [],
            'energy_consumption': [], 'comfort_violations': [], 'energy_cost': []
        }
        
        T_set_baseline = 23.0  # Fixed setpoint
        deadband = 1.0         # ±1°C deadband
        
        for t in range(duration):
            current_time = start_time + t
            if current_time >= len(building_data):
                break
                
            building_row = building_data.iloc[current_time]
            
            # Simple energy model
            T_actual = building_row['indoor_dry_bulb_temperature']
            cooling_demand = max(0, T_actual - (T_set_baseline + deadband)) * 2.0
            heating_demand = max(0, (T_set_baseline - deadband) - T_actual) * 2.0
            energy_consumption = (cooling_demand + heating_demand) / 3.0
            
            # Comfort violation check
            comfort_violation = 1 if (T_actual < 21.0 or T_actual > 25.0) else 0
            
            baseline_results['time'].append(current_time)
            baseline_results['indoor_temperature'].append(T_actual)
            baseline_results['indoor_humidity'].append(building_row['indoor_relative_humidity'])
            baseline_results['energy_consumption'].append(energy_consumption)
            baseline_results['comfort_violations'].append(comfort_violation)
            baseline_results['energy_cost'].append(energy_consumption * 0.12)
        
        return baseline_results

    def compare_mpc_vs_baseline(self, mpc_results, baseline_results):
        """Compare MPC vs baseline performance"""
        if not mpc_results['time'] or not baseline_results['time']:
            return {'cost_reduction_pct': 0, 'energy_reduction_pct': 0, 
                   'mpc_violation_rate': 0, 'baseline_violation_rate': 0}
        
        mpc_total_cost = sum(mpc_results['energy_cost'])
        mpc_violations = sum(mpc_results['comfort_violations'])
        mpc_avg_energy = np.mean(mpc_results['energy_consumption'])
        
        baseline_total_cost = sum(baseline_results['energy_cost'])
        baseline_violations = sum(baseline_results['comfort_violations'])
        baseline_avg_energy = np.mean(baseline_results['energy_consumption'])
        
        cost_reduction_pct = ((baseline_total_cost - mpc_total_cost) / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        energy_reduction_pct = ((baseline_avg_energy - mpc_avg_energy) / baseline_avg_energy * 100) if baseline_avg_energy > 0 else 0
        
        return {
            'cost_reduction_pct': cost_reduction_pct,
            'energy_reduction_pct': energy_reduction_pct,
            'mpc_violation_rate': mpc_violations / len(mpc_results['time']) * 100,
            'baseline_violation_rate': baseline_violations / len(baseline_results['time']) * 100,
            'mpc_total_cost': mpc_total_cost,
            'baseline_total_cost': baseline_total_cost
        }

    def calculate_annual_summary(self, season_metrics, season_results):
        """Calculate annual performance summary"""
        if not season_metrics:
            return {}
        
        total_cost = sum(metrics['total_energy_cost'] for metrics in season_metrics.values())
        avg_violation_rate = np.mean([metrics['violation_rate'] for metrics in season_metrics.values()])
        avg_energy_consumption = np.mean([metrics['avg_energy_consumption'] for metrics in season_metrics.values()])
        avg_success_rate = np.mean([metrics['success_rate'] for metrics in season_metrics.values()])
        
        # Calculate total baseline savings
        total_baseline_cost = 0
        total_mpc_cost = 0
        for season_data in season_results.values():
            if 'comparison' in season_data:
                total_baseline_cost += season_data['comparison']['baseline_total_cost']
                total_mpc_cost += season_data['comparison']['mpc_total_cost']
        
        annual_savings_pct = ((total_baseline_cost - total_mpc_cost) / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        
        return {
            'total_annual_cost': total_cost,
            'avg_violation_rate': avg_violation_rate,
            'avg_energy_consumption': avg_energy_consumption,
            'avg_success_rate': avg_success_rate,
            'annual_savings_vs_baseline': annual_savings_pct,
            'number_of_seasons': len(season_metrics)
        }

    def run_comprehensive_analysis(self):
        """Run complete analysis across all combinations"""
        print("COMPREHENSIVE MPC ANALYSIS")
        print("="*80)
        print("Analyzing all building types across all weather conditions...")
        
        # Load all data
        data = self.load_and_prepare_data()
        if data is None:
            return
        
        # Run simulations for all combinations
        total_combinations = len(data['buildings']) * len(data['weather'])
        current_combination = 0
        
        for i, building_data in enumerate(data['buildings']):
            for j, weather_data in enumerate(data['weather']):
                current_combination += 1
                building_name = data['building_names'][i]
                weather_name = data['weather_names'][j]
                
                print(f"\nProgress: {current_combination}/{total_combinations}")
                
                try:
                    annual_summary = self.run_single_simulation(
                        building_data, weather_data, building_name, weather_name
                    )
                    
                except Exception as e:
                    print(f"Error in {building_name}-{weather_name} simulation: {str(e)}")
                    continue
        
        # Generate comprehensive analysis
        self.generate_comprehensive_analysis()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"Results saved in: {self.output_dir}")
        print(f"{'='*80}")

    def generate_comprehensive_analysis(self):
        """Generate overall analysis across all combinations"""
        print("\nGenerating comprehensive analysis...")
        
        # Create performance matrix
        self.chart_generator.create_performance_matrix(self.results_summary)
        
        # Create comparative analysis
        self.statistical_analyzer.create_comparative_analysis(self.results_summary, self.output_dir)
        
        # Create specialized analyses
        self.statistical_analyzer.create_climate_impact_analysis(self.results_summary, self.output_dir)
        self.statistical_analyzer.create_building_performance_analysis(self.results_summary, self.output_dir)
        self.statistical_analyzer.create_seasonal_trends_analysis(self.results_summary, self.output_dir)
        self.statistical_analyzer.create_rl_benchmark_recommendations(self.results_summary, self.output_dir)
        
        # Save detailed results
        self.data_manager.save_detailed_results(self.results_summary, self.output_dir)
        
        print("Comprehensive analysis complete!")

    def generate_executive_summary(self):
        """Generate executive summary report"""
        self.report_generator.generate_executive_summary(self.results_summary)