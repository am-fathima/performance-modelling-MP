"""
CLINIC SIMULATION VISUALIZATIONS
Generates graphs and tables for clinic queuing analysis

Visualizations include:
1. Queue length versus time graphs showing congestion patterns
2. Comparative charts illustrating waiting times under different doctor configurations
3. Simulation-based performance tables summarizing key metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
from sim_outpatient_clinic import OneDayClinicSimulation

class ClinicVisualizations:
    def __init__(self):
        """Initialize visualization class"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'accent1': '#d62728',
            'accent2': '#9467bd',
            'accent3': '#8c564b',
            'accent4': '#e377c2'
        }
    
    def plot_queue_length_over_time(self, clinic_sim, save_path=None):
        """
        Plot queue length versus time showing congestion patterns
        
        Args:
            clinic_sim: OneDayClinicSimulation object with simulation results
            save_path: Optional path to save the figure
        """
        if not clinic_sim.queue_lengths:
            print("No queue length data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Queue Length Analysis Over Time', fontsize=16, fontweight='bold')
        
        # 1. Queue length over entire day
        ax1 = axes[0, 0]
        time_points = range(len(clinic_sim.queue_lengths))
        ax1.plot(time_points, clinic_sim.queue_lengths, 
                color=self.colors['primary'], linewidth=2)
        ax1.axvline(x=180, color='red', linestyle='--', alpha=0.7, 
                   label='Doctors Start (9:00 AM)')
        ax1.axvline(x=420, color='green', linestyle='--', alpha=0.7, 
                   label='Clinic Closes (1:00 PM)')
        ax1.set_xlabel('Time (minutes from 6:00 AM)', fontsize=10)
        ax1.set_ylabel('Queue Length', fontsize=10)
        ax1.set_title('Queue Length Throughout the Day', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight congestion periods
        high_congestion = [i for i, q in enumerate(clinic_sim.queue_lengths) if q > np.mean(clinic_sim.queue_lengths) * 1.5]
        if high_congestion:
            ax1.fill_between(high_congestion, 0, 
                           [clinic_sim.queue_lengths[i] for i in high_congestion],
                           color='red', alpha=0.2, label='High Congestion')
            ax1.legend()
        
        # 2. Queue length distribution
        ax2 = axes[0, 1]
        unique_queues, counts = np.unique(clinic_sim.queue_lengths, return_counts=True)
        ax2.bar(unique_queues, counts, 
               color=self.colors['secondary'], alpha=0.7)
        ax2.set_xlabel('Queue Length', fontsize=10)
        ax2.set_ylabel('Frequency (minutes)', fontsize=10)
        ax2.set_title('Distribution of Queue Lengths', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        avg_queue = np.mean(clinic_sim.queue_lengths)
        max_queue = np.max(clinic_sim.queue_lengths)
        ax2.axvline(x=avg_queue, color='red', linestyle='--', 
                   label=f'Mean: {avg_queue:.1f}')
        ax2.axvline(x=np.median(clinic_sim.queue_lengths), color='green', 
                   linestyle='--', label=f'Median: {np.median(clinic_sim.queue_lengths):.1f}')
        ax2.legend()
        
        # 3. Queue length by hour
        ax3 = axes[1, 0]
        hours = ['6-7 AM', '7-8 AM', '8-9 AM', '9-10 AM', '10-11 AM', '11-12 PM', '12-1 PM']
        hourly_queues = []
        
        for hour in range(7):  # 7 hours from 6 AM to 1 PM
            start_min = hour * 60
            end_min = (hour + 1) * 60
            hour_queues = clinic_sim.queue_lengths[start_min:end_min]
            hourly_queues.append(np.mean(hour_queues) if hour_queues else 0)
        
        bars = ax3.bar(hours, hourly_queues, 
                      color=[self.colors['primary'] if i < 3 else self.colors['accent1'] 
                            for i in range(7)], alpha=0.7)
        
        # Color before and after doctors start differently
        for i in range(3):
            bars[i].set_color(self.colors['tertiary'])
        
        ax3.set_xlabel('Hour of Day', fontsize=10)
        ax3.set_ylabel('Average Queue Length', fontsize=10)
        ax3.set_title('Average Queue Length by Hour', fontsize=12)
        ax3.set_xticklabels(hours, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, hourly_queues):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Cumulative patients served
        ax4 = axes[1, 1]
        if clinic_sim.patient_data:
            patient_times = [p['service_start'] for p in clinic_sim.patient_data]
            patient_times_sorted = sorted(patient_times)
            patients_served = list(range(1, len(patient_times_sorted) + 1))
            
            ax4.plot(patient_times_sorted, patients_served, 
                    color=self.colors['accent2'], linewidth=2, marker='o', markersize=3)
            ax4.axvline(x=180, color='red', linestyle='--', alpha=0.7, 
                       label='Doctors Start (9:00 AM)')
            ax4.set_xlabel('Time (minutes from 6:00 AM)', fontsize=10)
            ax4.set_ylabel('Cumulative Patients Served', fontsize=10)
            ax4.set_title('Cumulative Patients Served Over Time', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_doctor_configurations(self, patients_per_hour=30, save_path=None):
        """
        Compare waiting times under different doctor configurations
        
        Args:
            patients_per_hour: Average patients arriving per hour
            save_path: Optional path to save the figure
        """
        doctor_configs = [1, 2, 3, 4, 5]
        results = []
        
        print(f"\n{'='*70}")
        print("RUNNING SIMULATIONS FOR DIFFERENT DOCTOR CONFIGURATIONS")
        print(f"{'='*70}")
        
        for num_doctors in doctor_configs:
            print(f"Running simulation with {num_doctors} doctor(s)...")
            
            clinic = OneDayClinicSimulation(
                num_doctors=num_doctors,
                patients_per_hour=patients_per_hour
            )
            
            clinic.simulate_day()
            
            if clinic.waiting_times:
                results.append({
                    'doctors': num_doctors,
                    'avg_wait': np.mean(clinic.waiting_times),
                    'max_wait': np.max(clinic.waiting_times),
                    'patients_served': len(clinic.waiting_times),
                    'avg_queue': np.mean(clinic.queue_lengths),
                    'max_queue': np.max(clinic.queue_lengths),
                    'utilization': (sum(clinic.service_times) / 
                                  (num_doctors * 240)) * 100 if clinic.service_times else 0
                })
        
        # Create DataFrame for results
        df_results = pd.DataFrame(results)
        
        # Create comparative visualizations
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Average waiting time comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(df_results['doctors'].astype(str), df_results['avg_wait'],
                       color=[self.colors['primary'], self.colors['secondary'], 
                             self.colors['tertiary'], self.colors['accent1'], 
                             self.colors['accent2']],
                       alpha=0.7)
        ax1.set_xlabel('Number of Doctors', fontsize=10)
        ax1.set_ylabel('Average Waiting Time (minutes)', fontsize=10)
        ax1.set_title('Average Waiting Time by Doctor Count', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars1, df_results['avg_wait']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Add target line (30 minutes)
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, 
                   label='Target: 30 minutes')
        ax1.legend()
        
        # 2. Maximum waiting time comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(df_results['doctors'].astype(str), df_results['max_wait'],
                       color=[self.colors['accent1'] for _ in range(len(df_results))],
                       alpha=0.7)
        ax2.set_xlabel('Number of Doctors', fontsize=10)
        ax2.set_ylabel('Maximum Waiting Time (minutes)', fontsize=10)
        ax2.set_title('Maximum Waiting Time by Doctor Count', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, df_results['max_wait']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Doctor utilization comparison
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(df_results['doctors'].astype(str), df_results['utilization'],
                       color=[self.colors['secondary'] for _ in range(len(df_results))],
                       alpha=0.7)
        ax3.set_xlabel('Number of Doctors', fontsize=10)
        ax3.set_ylabel('Utilization Rate (%)', fontsize=10)
        ax3.set_title('Doctor Utilization Rate by Doctor Count', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Capacity')
        ax3.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Optimal: 80%')
        
        for bar, value in zip(bars3, df_results['utilization']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        ax3.legend()
        
        # 4. Patients served comparison
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(df_results['doctors'].astype(str), df_results['patients_served'],
                       color=[self.colors['tertiary'] for _ in range(len(df_results))],
                       alpha=0.7)
        ax4.set_xlabel('Number of Doctors', fontsize=10)
        ax4.set_ylabel('Patients Served', fontsize=10)
        ax4.set_title('Total Patients Served by Doctor Count', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars4, df_results['patients_served']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom', fontsize=9)
        
        # 5. Average queue length comparison
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(df_results['doctors'].astype(str), df_results['avg_queue'],
                       color=[self.colors['accent2'] for _ in range(len(df_results))],
                       alpha=0.7)
        ax5.set_xlabel('Number of Doctors', fontsize=10)
        ax5.set_ylabel('Average Queue Length', fontsize=10)
        ax5.set_title('Average Queue Length by Doctor Count', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars5, df_results['avg_queue']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Waiting time reduction analysis
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Calculate percentage reduction from baseline (1 doctor)
        if len(df_results) > 1:
            baseline_wait = df_results.loc[0, 'avg_wait']
            wait_reductions = []
            for idx, row in df_results.iterrows():
                reduction = ((baseline_wait - row['avg_wait']) / baseline_wait) * 100
                wait_reductions.append(reduction)
            
            bars6 = ax6.bar(df_results['doctors'].astype(str), wait_reductions,
                          color=[self.colors['accent3'] for _ in range(len(df_results))],
                          alpha=0.7)
            ax6.set_xlabel('Number of Doctors', fontsize=10)
            ax6.set_ylabel('Waiting Time Reduction (%)', fontsize=10)
            ax6.set_title('Waiting Time Reduction vs 1 Doctor Baseline', fontsize=12)
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars6, wait_reductions):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Comparative Analysis: Clinic Performance by Doctor Configuration\n'
                    f'Arrival Rate: {patients_per_hour} patients/hour', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return results for table generation
        return df_results
    
    def create_performance_table(self, df_results, save_path=None):
        """
        Create simulation-based performance table summarizing key metrics
        
        Args:
            df_results: DataFrame with simulation results
            save_path: Optional path to save the table as image
        """
        # Create a formatted table
        table_data = df_results.copy()
        
        # Format the values
        table_data['avg_wait'] = table_data['avg_wait'].apply(lambda x: f'{x:.1f} min')
        table_data['max_wait'] = table_data['max_wait'].apply(lambda x: f'{x:.1f} min')
        table_data['avg_queue'] = table_data['avg_queue'].apply(lambda x: f'{x:.1f}')
        table_data['max_queue'] = table_data['max_queue'].apply(lambda x: f'{x:.0f}')
        table_data['utilization'] = table_data['utilization'].apply(lambda x: f'{x:.1f}%')
        
        # Rename columns for better display
        table_data = table_data.rename(columns={
            'doctors': 'Doctors',
            'avg_wait': 'Avg Wait',
            'max_wait': 'Max Wait',
            'patients_served': 'Patients',
            'avg_queue': 'Avg Queue',
            'max_queue': 'Max Queue',
            'utilization': 'Utilization'
        })
        
        # Create matplotlib table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=[self.colors['primary']] * len(table_data.columns))
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Color code cells based on values
        for i in range(len(table_data)):
            avg_wait = float(table_data.iloc[i]['Avg Wait'].split()[0])
            if avg_wait < 30:
                table[(i+1, 1)].set_facecolor('#90EE90')  # Light green
            elif avg_wait < 60:
                table[(i+1, 1)].set_facecolor('#FFD700')  # Yellow
            else:
                table[(i+1, 1)].set_facecolor('#FFB6C1')  # Light red
            
            utilization = float(table_data.iloc[i]['Utilization'].replace('%', ''))
            if 70 <= utilization <= 90:
                table[(i+1, 6)].set_facecolor('#90EE90')  # Light green
            elif utilization > 90:
                table[(i+1, 6)].set_facecolor('#FFD700')  # Yellow
            else:
                table[(i+1, 6)].set_facecolor('#FFB6C1')  # Light red
        
        plt.title('Performance Metrics by Doctor Configuration\n', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add summary statistics
        summary_text = f"""
        Key Insights:
        1. Adding each additional doctor reduces average waiting time significantly
        2. Optimal utilization (70-90%) achieved with {df_results.loc[df_results['utilization'].between(70, 90)].shape[0]} doctor configurations
        3. Target wait time (<30 min) achieved with {df_results.loc[df_results['avg_wait'] < 30].shape[0]} doctor configurations
        4. Maximum queue reduction: {df_results['max_queue'].max() - df_results['max_queue'].min()} patients
        """
        
        plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also print the table in console
        print("\n" + "="*70)
        print("PERFORMANCE METRICS SUMMARY TABLE")
        print("="*70)
        print(table_data.to_string(index=False))
        
        return table_data
    
    def create_waiting_time_distribution(self, clinic_sim, save_path=None):
        """
        Create detailed waiting time distribution visualization
        
        Args:
            clinic_sim: OneDayClinicSimulation object
            save_path: Optional path to save the figure
        """
        if not clinic_sim.waiting_times:
            print("No waiting time data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Waiting Time Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram of waiting times
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(clinic_sim.waiting_times, bins=30, 
                                   color=self.colors['primary'], 
                                   edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Waiting Time (minutes)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Distribution of Patient Waiting Times', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics lines
        mean_wait = np.mean(clinic_sim.waiting_times)
        median_wait = np.median(clinic_sim.waiting_times)
        ax1.axvline(mean_wait, color='red', linestyle='--', 
                   label=f'Mean: {mean_wait:.1f} min')
        ax1.axvline(median_wait, color='green', linestyle='--', 
                   label=f'Median: {median_wait:.1f} min')
        ax1.legend()
        
        # 2. Cumulative distribution function
        ax2 = axes[0, 1]
        sorted_wait = np.sort(clinic_sim.waiting_times)
        cdf = np.arange(1, len(sorted_wait) + 1) / len(sorted_wait)
        ax2.plot(sorted_wait, cdf, 
                color=self.colors['accent1'], linewidth=2)
        ax2.set_xlabel('Waiting Time (minutes)', fontsize=10)
        ax2.set_ylabel('Cumulative Probability', fontsize=10)
        ax2.set_title('Cumulative Distribution of Waiting Times', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add percentile markers
        percentiles = [50, 75, 90, 95]
        for p in percentiles:
            value = np.percentile(clinic_sim.waiting_times, p)
            ax2.axvline(value, color='gray', linestyle=':', alpha=0.5)
            ax2.text(value, 0.05, f'{p}%: {value:.1f} min', 
                    rotation=90, fontsize=9, verticalalignment='bottom')
        
        # 3. Box plot of waiting times
        ax3 = axes[1, 0]
        bp = ax3.boxplot(clinic_sim.waiting_times, patch_artist=True,
                        boxprops=dict(facecolor=self.colors['secondary']),
                        medianprops=dict(color='black'))
        ax3.set_ylabel('Waiting Time (minutes)', fontsize=10)
        ax3.set_title('Box Plot of Waiting Times', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = f"""
        Statistics:
        Min: {np.min(clinic_sim.waiting_times):.1f} min
        Q1: {np.percentile(clinic_sim.waiting_times, 25):.1f} min
        Median: {median_wait:.1f} min
        Q3: {np.percentile(clinic_sim.waiting_times, 75):.1f} min
        Max: {np.max(clinic_sim.waiting_times):.1f} min
        IQR: {np.percentile(clinic_sim.waiting_times, 75) - np.percentile(clinic_sim.waiting_times, 25):.1f} min
        """
        ax3.text(1.1, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Waiting time vs arrival time
        ax4 = axes[1, 1]
        if clinic_sim.patient_data:
            arrival_times = [p['arrival'] for p in clinic_sim.patient_data]
            waiting_times = [p['waiting_time'] for p in clinic_sim.patient_data]
            
            scatter = ax4.scatter(arrival_times, waiting_times,
                                c=waiting_times, cmap='viridis',
                                alpha=0.6, s=50)
            ax4.axvline(x=180, color='red', linestyle='--', alpha=0.7,
                       label='Doctors Start (9:00 AM)')
            ax4.set_xlabel('Arrival Time (minutes from 6:00 AM)', fontsize=10)
            ax4.set_ylabel('Waiting Time (minutes)', fontsize=10)
            ax4.set_title('Waiting Time vs Arrival Time', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax4, label='Waiting Time (minutes)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*70)
    print("CLINIC SIMULATION VISUALIZATIONS GENERATOR")
    print("="*70)
    
    # Initialize visualization class
    viz = ClinicVisualizations()
    
    # Run base simulation with 3 doctors
    print("\n1. Running base simulation with 3 doctors...")
    base_clinic = OneDayClinicSimulation(num_doctors=3, patients_per_hour=30)
    base_clinic.simulate_day()
    
    # Generate all visualizations
    print("\n2. Generating queue length vs time graphs...")
    viz.plot_queue_length_over_time(base_clinic, save_path="queue_length_analysis.png")
    
    print("\n3. Generating waiting time distribution analysis...")
    viz.create_waiting_time_distribution(base_clinic, save_path="waiting_time_distribution.png")
    
    print("\n4. Comparing different doctor configurations...")
    df_results = viz.compare_doctor_configurations(patients_per_hour=30, 
                                                   save_path="doctor_comparison.png")
    
    print("\n5. Creating performance summary table...")
    viz.create_performance_table(df_results, save_path="performance_table.png")
    
    print("\n" + "="*70)
    print("VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("1. queue_length_analysis.png - Queue length patterns throughout the day")
    print("2. waiting_time_distribution.png - Detailed waiting time analysis")
    print("3. doctor_comparison.png - Performance comparison by doctor count")
    print("4. performance_table.png - Summary table of key metrics")
    print("\nThese visualizations demonstrate how early arrivals and limited capacity affect system performance.")


if __name__ == "__main__":
    # Note: Make sure one_day_clinic_simulation.py is in the same directory
    # or adjust the import accordingly
    main()