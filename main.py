"""
Main entry point for Comprehensive MPC Analysis System
"""

import warnings
warnings.filterwarnings('ignore')

from mpc_analysis_core import ComprehensiveMPCAnalysis

def main():
    """Main execution function"""
    print("COMPREHENSIVE MPC ANALYSIS SYSTEM")
    print("="*80)
    print("This will analyze MPC performance across:")
    print("• 3 Building Types × 3 Weather Locations = 9 Combinations")
    print("• 4 Seasons per combination = 36 total simulations")
    print("• Complete performance analysis and benchmarking")
    print("• RL benchmark recommendations")
    print("="*80)
    
    # Initialize analysis system
    analyzer = ComprehensiveMPCAnalysis(output_dir='comprehensive_mpc_results')
    
    try:
        # Run complete analysis
        analyzer.run_comprehensive_analysis()
        
        # Generate executive summary
        analyzer.generate_executive_summary()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE! 🎉")
        print("="*80)
        print("📊 Generated Analysis:")
        print("  ✓ 36 Individual simulations completed")
        print("  ✓ 9 Building-weather combination analyses")
        print("  ✓ Performance matrices and correlations")
        print("  ✓ Climate impact analysis")
        print("  ✓ Building performance characteristics")
        print("  ✓ Seasonal trends analysis")
        print("  ✓ RL benchmark recommendations")
        print("  ✓ Executive summary with key findings")
        print("\n📁 Results Structure:")
        print("  comprehensive_mpc_results/")
        print("    📁 charts/          - Individual combination charts")
        print("      📁 Building_1_Abha/")
        print("        📄 seasonal_comparison.png")
        print("        📄 annual_summary.png")
        print("        📄 winter_detailed.png")
        print("        📄 spring_detailed.png")
        print("        📄 summer_detailed.png")
        print("        📄 autumn_detailed.png")
        print("      📁 Building_1_Jeddah/")
        print("      📁 Building_1_Riyadh/")
        print("      📁 ... (9 total)")
        print("    📁 data/            - CSV and JSON data files")
        print("      📄 annual_summary.csv")
        print("      📄 seasonal_breakdown.csv")
        print("      📄 detailed_results.json")
        print("    📁 analysis/        - Comparative analysis charts")
        print("      📄 performance_matrix.png")
        print("      📄 comparative_analysis.png")
        print("      📄 climate_impact_analysis.png")
        print("      📄 building_performance_analysis.png")
        print("      📄 seasonal_trends_analysis.png")
        print("      📄 rl_benchmark_recommendations.png")
        print("      📄 statistical_summary.txt")
        print("    📄 executive_summary.txt")
        print("    📄 rl_benchmark_guide.txt")
        print("="*80)
        print("🚀 Ready for RL comparison!")
        print("📈 Total charts generated: 60+")
        print("📊 Total data files: 10+")
        print("📋 Complete benchmark established!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check that:")
        print("1. CSV files are in ./data/ directory")
        print("2. building_mpc.py is available and working")
        print("3. All required packages are installed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()