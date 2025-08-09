"""
Data Manager for MPC Analysis System
"""

import pandas as pd
import json

class DataManager:
    """
    Handles data loading, validation, and saving operations
    """
    
    def __init__(self):
        pass
    
    def load_and_prepare_data(self):
        """Load and prepare all CSV data"""
        print("Loading building and weather data...")
        
        try:
            # Load building datasets
            building_1 = pd.read_csv('./data/building_1.csv')
            building_2 = pd.read_csv('./data/building_2.csv') 
            building_3 = pd.read_csv('./data/building_3.csv')
            
            # Load weather datasets
            weather_abha = pd.read_csv('./data/weather_abha.csv')
            weather_jeddah = pd.read_csv('./data/weather_jeddah.csv')
            weather_riyadh = pd.read_csv('./data/weather_riyadh.csv')
            
            print("Data loaded successfully!")
            print(f"Building data shapes: {building_1.shape}, {building_2.shape}, {building_3.shape}")
            print(f"Weather data shapes: {weather_abha.shape}, {weather_jeddah.shape}, {weather_riyadh.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure CSV files are in ./data/ directory")
            return None
        
        # Data cleaning and validation
        buildings = [building_1, building_2, building_3]
        weather_data = [weather_abha, weather_jeddah, weather_riyadh]
        
        for i, building in enumerate(buildings):
            buildings[i] = building.fillna(method='ffill').fillna(method='bfill')
            
        for i, weather in enumerate(weather_data):
            weather_data[i] = weather.fillna(method='ffill').fillna(method='bfill')
        
        # Validate data ranges
        for i, building in enumerate(buildings):
            temp_range = building['indoor_dry_bulb_temperature']
            humidity_range = building['indoor_relative_humidity']
            print(f"Building {i+1} - Temperature: {temp_range.min():.1f} to {temp_range.max():.1f}°C, "
                  f"Humidity: {humidity_range.min():.1f} to {humidity_range.max():.1f}%")
        
        return {
            'buildings': buildings,
            'weather': weather_data,
            'building_names': ['Building_1', 'Building_2', 'Building_3'],
            'weather_names': ['Abha', 'Jeddah', 'Riyadh']
        }
    
    def save_detailed_results(self, results_summary, output_dir):
        """Save detailed results to CSV files"""
        print("Saving detailed results...")
        
        # Prepare summary data
        summary_data = []
        seasonal_data = []
        
        for combo_key, data in results_summary.items():
            building, weather = combo_key.split('_')
            annual = data['annual']
            
            # Annual summary
            summary_data.append({
                'Building': building,
                'Weather': weather,
                'Annual_Cost': annual.get('total_annual_cost', 0),
                'Avg_Violation_Rate': annual.get('avg_violation_rate', 0),
                'Avg_Energy_Consumption': annual.get('avg_energy_consumption', 0),
                'Annual_Savings': annual.get('annual_savings_vs_baseline', 0),
                'Success_Rate': annual.get('avg_success_rate', 0),
                'Seasons_Analyzed': annual.get('number_of_seasons', 0)
            })
            
            # Seasonal breakdown
            for season, metrics in data['seasons'].items():
                seasonal_data.append({
                    'Building': building,
                    'Weather': weather,
                    'Season': season,
                    'Cost': metrics.get('total_energy_cost', 0),
                    'Violation_Rate': metrics.get('violation_rate', 0),
                    'Avg_Energy': metrics.get('avg_energy_consumption', 0),
                    'Success_Rate': metrics.get('success_rate', 0)
                })
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        seasonal_df = pd.DataFrame(seasonal_data)
        
        summary_df.to_csv(f'{output_dir}/data/annual_summary.csv', index=False)
        seasonal_df.to_csv(f'{output_dir}/data/seasonal_breakdown.csv', index=False)
        
        # Save detailed results as JSON for potential future analysis
        with open(f'{output_dir}/data/detailed_results.json', 'w') as f:
            # Remove detailed time series data for JSON (too large)
            simplified_results = {}
            for combo_key, data in results_summary.items():
                simplified_results[combo_key] = {
                    'building': data['building'],
                    'weather': data['weather'],
                    'annual': data['annual'],
                    'seasons': data['seasons']
                }
            json.dump(simplified_results, f, indent=2)
        
        print(f"Detailed results saved to {output_dir}/data/")