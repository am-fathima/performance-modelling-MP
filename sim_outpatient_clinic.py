"""
ONE-DAY SIMULATION FOR MONTHLY GOVERNMENT HOSPITAL CLINIC
M/M/c Queuing Model

System Scenario:
- Patients start arriving at 6:00 AM for tokens
- Doctors work from 9:00 AM to 1:00 PM (4 hours)
- Service time: 3-10 minutes per patient
- Queuing model: M/M/c (Poisson arrivals, Exponential service)

Performance Objective:
Minimize average patient waiting time
"""

import heapq
import random
import numpy as np
from datetime import datetime, timedelta

class OneDayClinicSimulation:
    def __init__(self, num_doctors=3, patients_per_hour=30, seed=42):
        """
        Initialize one-day clinic simulation
        
        Parameters:
        - num_doctors: Number of doctors (c)
        - patients_per_hour: Average patients arriving per hour (λ)
        - seed: Random seed for reproducibility
        """
        self.num_doctors = num_doctors
        self.lambda_rate = patients_per_hour / 60  # Patients per minute
        self.mu_rate = 1/6.5  # Service rate (average 6.5 minutes per patient)
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Clinic timings (in minutes from 6:00 AM)
        self.clinic_open = 0      # 6:00 AM
        self.doctors_start = 180  # 9:00 AM (180 minutes from 6 AM)
        self.clinic_close = 420   # 1:00 PM (420 minutes from 6 AM)
        
        # Statistics
        self.waiting_times = []
        self.queue_lengths = []
        self.service_times = []
        self.patient_data = []
        
    def simulate_day(self):
        """Run one-day simulation"""
        print("\n" + "="*70)
        print("ONE-DAY CLINIC SIMULATION")
        print("="*70)
        
        # Initialize simulation
        current_time = 0
        events = []  # Priority queue for events
        queue = []   # Patient queue [(patient_id, arrival_time)]
        doctors_busy = 0
        patient_id = 0
        
        # Generate patient arrivals (6:00 AM to 1:00 PM)
        arrival_time = 0
        while arrival_time <= self.clinic_close:
            heapq.heappush(events, (arrival_time, 'arrival', patient_id))
            
            # Generate next arrival time (exponential distribution)
            interarrival = np.random.exponential(1/self.lambda_rate)
            arrival_time += interarrival
            patient_id += 1
        
        total_patients = patient_id
        
        # Track patient information
        patient_arrival_times = {}
        patient_service_start_times = {}
        patient_service_end_times = {}
        
        print(f"\nClinic opens at: 6:00 AM")
        print(f"Doctors start at: 9:00 AM")
        print(f"Clinic closes at: 1:00 PM")
        print(f"Number of doctors: {self.num_doctors}")
        print(f"Total patients arriving: {total_patients}")
        print(f"Average arrival rate: {self.lambda_rate*60:.1f} patients/hour")
        print("\nSimulation running...")
        
        # Process events
        minute_counter = 0
        while events:
            current_time, event_type, pid = heapq.heappop(events)
            
            # Record queue length every minute
            while minute_counter <= current_time:
                self.queue_lengths.append(len(queue))
                minute_counter += 1
            
            if event_type == 'arrival':
                patient_arrival_times[pid] = current_time
                queue.append((pid, current_time))
                
                # Start service if doctors available
                if doctors_busy < self.num_doctors and queue and current_time >= self.doctors_start:
                    self.start_service(current_time, queue, doctors_busy, events, 
                                     patient_arrival_times, patient_service_start_times)
                    doctors_busy += 1
            
            elif event_type == 'start_service':
                if pid in patient_arrival_times:
                    patient_service_start_times[pid] = current_time
                    
                    # Calculate waiting time
                    waiting_time = current_time - patient_arrival_times[pid]
                    self.waiting_times.append(waiting_time)
                    
                    # Generate service time (3-10 minutes)
                    service_time = np.random.uniform(3, 10)
                    self.service_times.append(service_time)
                    
                    # Schedule departure
                    departure_time = current_time + service_time
                    heapq.heappush(events, (departure_time, 'departure', pid))
                    
                    # Record patient data
                    self.patient_data.append({
                        'id': pid,
                        'arrival': patient_arrival_times[pid],
                        'service_start': current_time,
                        'waiting_time': waiting_time,
                        'service_time': service_time,
                        'departure': departure_time
                    })
            
            elif event_type == 'departure':
                doctors_busy -= 1
                if pid in patient_arrival_times:
                    patient_service_end_times[pid] = current_time
                
                # Start next patient if available
                if queue and doctors_busy < self.num_doctors and current_time >= self.doctors_start:
                    self.start_service(current_time, queue, doctors_busy, events,
                                     patient_arrival_times, patient_service_start_times)
                    doctors_busy += 1
        
        # Calculate statistics
        self.calculate_statistics(total_patients)
        
        return self.patient_data
    
    def start_service(self, current_time, queue, doctors_busy, events, 
                     patient_arrival_times, patient_service_start_times):
        """Start service for next patient in queue"""
        if queue:
            pid, arrival_time = queue.pop(0)
            heapq.heappush(events, (current_time, 'start_service', pid))
    
    def calculate_statistics(self, total_patients):
        """Calculate and display simulation statistics"""
        patients_served = len(self.waiting_times)
        patients_not_served = total_patients - patients_served
        
        if patients_served > 0:
            avg_wait = np.mean(self.waiting_times)
            max_wait = np.max(self.waiting_times)
            avg_service = np.mean(self.service_times)
            avg_queue = np.mean(self.queue_lengths)
            max_queue = np.max(self.queue_lengths)
            
            # Doctor utilization
            total_service_time = sum(self.service_times)
            available_time = self.num_doctors * (self.clinic_close - self.doctors_start)
            utilization = (total_service_time / available_time) * 100
            
            print("\n" + "="*70)
            print("SIMULATION RESULTS")
            print("="*70)
            
            print(f"\nPATIENT STATISTICS:")
            print(f"  Total patients arrived: {total_patients}")
            print(f"  Patients served: {patients_served}")
            print(f"  Patients not served: {patients_not_served}")
            print(f"  Service completion rate: {(patients_served/total_patients)*100:.1f}%")
            
            print(f"\nWAITING TIME STATISTICS:")
            print(f"  Average waiting time: {avg_wait:.2f} minutes")
            print(f"  Maximum waiting time: {max_wait:.2f} minutes")
            print(f"  Average waiting time: {avg_wait/60:.2f} hours")
            
            print(f"\nSERVICE TIME STATISTICS:")
            print(f"  Average service time: {avg_service:.2f} minutes")
            print(f"  Service time range: 3.0 - 10.0 minutes")
            
            print(f"\nQUEUE STATISTICS:")
            print(f"  Average queue length: {avg_queue:.2f} patients")
            print(f"  Maximum queue length: {max_queue} patients")
            
            print(f"\nDOCTOR UTILIZATION:")
            print(f"  Doctor utilization: {utilization:.1f}%")
            
            # Waiting time distribution
            print(f"\nWAITING TIME DISTRIBUTION:")
            thresholds = [15, 30, 60, 120, 180]
            for threshold in thresholds:
                count = sum(1 for wt in self.waiting_times if wt > threshold)
                percentage = (count / patients_served) * 100
                print(f"  Waiting > {threshold} min: {count} patients ({percentage:.1f}%)")
            
            # Time analysis
            print(f"\nTIME ANALYSIS:")
            first_patient_arrival = min([p['arrival'] for p in self.patient_data]) if self.patient_data else 0
            last_patient_departure = max([p['departure'] for p in self.patient_data]) if self.patient_data else 0
            
            print(f"  First patient arrived at: {self.minutes_to_time(first_patient_arrival)}")
            print(f"  Last patient departed at: {self.minutes_to_time(last_patient_departure)}")
            print(f"  Total clinic operation: {(last_patient_departure - first_patient_arrival)/60:.1f} hours")
            
            # Performance assessment
            print(f"\nPERFORMANCE ASSESSMENT:")
            if avg_wait < 30:
                print(f"  ✓ GOOD: Average wait < 30 minutes")
            elif avg_wait < 60:
                print(f"  ⚠ FAIR: Average wait 30-60 minutes")
            else:
                print(f"  ✗ POOR: Average wait > 60 minutes")
                
            if utilization > 90:
                print(f"  ⚠ HIGH: Doctor utilization > 90%")
            elif utilization > 70:
                print(f"  ✓ OPTIMAL: Doctor utilization 70-90%")
            else:
                print(f"  ⚠ LOW: Doctor utilization < 70%")
        else:
            print("\nNo patients were served!")
    
    def minutes_to_time(self, minutes):
        """Convert minutes from 6:00 AM to time string"""
        total_minutes = 6 * 60 + minutes  # Start from 6:00 AM
        hours = total_minutes // 60
        mins = total_minutes % 60
        am_pm = "AM" if hours < 12 else "PM"
        hours = hours if hours <= 12 else hours - 12
        return f"{int(hours):02d}:{int(mins):02d} {am_pm}"
    
    def print_patient_log(self, max_patients=20):
        """Print detailed log for first few patients"""
        if not self.patient_data:
            print("\nNo patient data available.")
            return
        
        print("\n" + "="*70)
        print("DETAILED PATIENT LOG (First 20 patients)")
        print("="*70)
        print(f"{'ID':<4} {'Arrival':<10} {'Service Start':<12} {'Wait (min)':<10} {'Service (min)':<12} {'Departure':<10}")
        print("-" * 70)
        
        for i, patient in enumerate(self.patient_data[:max_patients]):
            arrival_time = self.minutes_to_time(patient['arrival'])
            service_start = self.minutes_to_time(patient['service_start'])
            departure = self.minutes_to_time(patient['departure'])
            
            print(f"{patient['id']:<4} {arrival_time:<10} {service_start:<12} "
                  f"{patient['waiting_time']:<10.1f} {patient['service_time']:<12.1f} {departure:<10}")
    
    def theoretical_mmc_calculation(self):
        """Calculate theoretical M/M/c metrics"""
        c = self.num_doctors
        λ = self.lambda_rate * 60  # patients per hour
        μ = 60 / 6.5  # patients per hour (since average service is 6.5 minutes)
        ρ = λ / (c * μ)
        
        print("\n" + "="*70)
        print("THEORETICAL M/M/c CALCULATIONS")
        print("="*70)
        
        print(f"\nParameters:")
        print(f"  Number of servers (c): {c}")
        print(f"  Arrival rate (λ): {λ:.2f} patients/hour")
        print(f"  Service rate (μ): {μ:.2f} patients/hour/doctor")
        print(f"  Traffic intensity (ρ): {ρ:.3f}")
        print(f"  System stable: {'Yes' if ρ < 1 else 'No'}")
        
        if ρ >= 1:
            print("\n  ⚠ WARNING: System is unstable (ρ ≥ 1)")
            print("  Patients will accumulate indefinitely!")
            return
        
        # Calculate P0 (probability of 0 patients in system)
        sum_term = 0
        for n in range(c):
            sum_term += (c * ρ) ** n / self.factorial(n)
        
        P0 = 1 / (sum_term + ((c * ρ) ** c) / (self.factorial(c) * (1 - ρ)))
        
        # Calculate Lq (average number in queue)
        Lq = (P0 * (c * ρ) ** c * ρ) / (self.factorial(c) * (1 - ρ) ** 2)
        
        # Calculate Wq (average waiting time in queue)
        Wq = Lq / λ * 60  # Convert to minutes
        
        # Calculate L (average number in system)
        L = Lq + (λ / μ)
        
        # Calculate W (average time in system)
        W = Wq + (1/μ * 60)
        
        print(f"\nTheoretical Results:")
        print(f"  Probability system empty (P0): {P0:.3f}")
        print(f"  Average patients in queue (Lq): {Lq:.2f}")
        print(f"  Average waiting time (Wq): {Wq:.2f} minutes")
        print(f"  Average patients in system (L): {L:.2f}")
        print(f"  Average time in system (W): {W:.2f} minutes")
        
        # Compare with simulation
        if self.waiting_times:
            sim_avg_wait = np.mean(self.waiting_times)
            print(f"\nComparison with Simulation:")
            print(f"  Theoretical Wq: {Wq:.2f} minutes")
            print(f"  Simulated average wait: {sim_avg_wait:.2f} minutes")
            print(f"  Difference: {abs(Wq - sim_avg_wait):.2f} minutes")
    
    def factorial(self, n):
        """Calculate factorial"""
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


def main():
    """Run the one-day simulation"""
    print("\n" + "="*70)
    print("MONTHLY GOVERNMENT HOSPITAL CLINIC - ONE DAY SIMULATION")
    print("="*70)
    
    # Create simulation with 3 doctors
    clinic = OneDayClinicSimulation(
        num_doctors=3,
        patients_per_hour=30  # Average 30 patients per hour
    )
    
    # Run simulation
    patient_data = clinic.simulate_day()
    
    # Show detailed patient log
    clinic.print_patient_log(max_patients=20)
    
    # Show theoretical calculations
    clinic.theoretical_mmc_calculation()
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS TO REDUCE WAITING TIME")
    print("="*70)
    print("1. Add 1 more doctor (4 total): Would reduce waiting time significantly")
    print("2. Implement appointment slots: Stagger patient arrivals")
    print("3. Extend clinic hours: Add afternoon session (2 PM - 5 PM)")
    print("4. Improve efficiency: Target 5-minute average consultation time")
    print("5. Pre-consultation screening: Handle paperwork before doctor sees patient")
    print("="*70)
    
    # Test with different number of doctors
    print("\n" + "="*70)
    print("COMPARISON WITH DIFFERENT DOCTOR COUNTS")
    print("="*70)
    
    for doctors in [2, 3, 4, 5]:
        test_clinic = OneDayClinicSimulation(
            num_doctors=doctors,
            patients_per_hour=30
        )
        test_clinic.simulate_day()
        print("-" * 50)


if __name__ == "__main__":
    main()