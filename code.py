import math
import numpy as np
import matplotlib.pyplot as plt

# 1. m/m/c queuing model implementation 
# 1. M/M/c Queuing Model Implementation

def calculate_mmc_metrics(lambda_rate, mu_rate, c_servers):
    """
    Calculates key performance metrics for a stable M/M/c queuing system.
    Requires: lambda < c * mu for stability (lambda < c * mu is stability criterion [3]).
    
    Args:
        lambda_rate (float): Arrival rate (λ) (events per unit time) [2], [4].
        mu_rate (float): Service rate (μ) (events per unit time) [2], [4].
        c_servers (int): Number of parallel servers (c) [5], [6].
    # Calculate utilization (rho)
    rho = lambda_rate / (c_servers * mu_rate)

    Returns:
        tuple: (Utilization (ρ), Avg Queue Length (Lq), Avg Waiting Time (Wq))
    """
    rho = lambda_rate / (c_servers * mu_rate)  # utilization (ρ) [4]

    # check for system stability, required for steady-state analysis [3]
    # Check if the system is stable. If rho >= 1, the queue grows forever.
    if rho >= 1:
        # system is unstable (queue grows infinitely)
        return rho, float('inf'), float('inf')

    # calculate p0 (probability of zero customers in the system)
    # Calculate P0 (probability of 0 customers in the system)
    sum_terms = 0
    for n in range(c_servers):
        sum_terms += ((lambda_rate / mu_rate) ** n) / math.factorial(n)

    # erlang c formula component (p_c = probability that an arrival must wait)
    # Erlang C formula components
    erlang_c_numerator = ((lambda_rate / mu_rate) ** c_servers) / math.factorial(c_servers)
    erlang_c_denominator = (1 - rho)

    P0 = (sum_terms + (erlang_c_numerator / erlang_c_denominator)) ** -1

    # calculate lq (average number of customers waiting in the queue) [4]
    # Calculate Lq (average queue length)
    Lq = (erlang_c_numerator / erlang_c_denominator) * rho * P0 
    
    # calculate wq (average waiting time in queue) using little's law: lq = λ * wq [7], [8]
    # Calculate Wq (average waiting time) using Little's Law
    Wq = Lq / lambda_rate 

    return rho, Lq, Wq

# 2. input data (placeholders) 
# 2. Input Data

# define the independent m/m/c queues (skill groups) [9], [5], [6]
# note: replace these placeholder values with parameters calculated from your 'mp dataset'
# lambda (calls/min), mu (calls/min), c (agents)
# Input data for the different skill groups
# lambda = arrival rate, mu = service rate, c = number of agents
SKILL_GROUP_DATA = {
    'Billing': {'lambda': 0.8, 'mu': 0.1, 'c': 9},  # utilization target ~ 89% (high)
    'Tech Support': {'lambda': 1.0, 'mu': 0.1, 'c': 13}, # utilization target ~ 77% (good)
    'Sales': {'lambda': 0.5, 'mu': 0.1, 'c': 6}     # utilization target ~ 83% (slightly high)
    'Billing': {'lambda': 0.8, 'mu': 0.1, 'c': 9},
    'Tech Support': {'lambda': 1.0, 'mu': 0.1, 'c': 13},
    'Sales': {'lambda': 0.5, 'mu': 0.1, 'c': 6}
}

# --- 3. analysis and visualization ---
# --- 3. Analysis and Visualization ---

def analyze_and_visualize_performance(data):
    """Calculates metrics for each Skill Group and generates a Utilization chart."""
    
    skill_names = []
    utilization_rates = []
    queue_lengths = []
    wait_times_secs = []
    
    print("\n--- PERFORMANCE METRICS PER SKILL GROUP ---")
    for skill, params in data.items():
        L = params['lambda']
        M = params['mu']
        C = params['c']
        
        rho, Lq, Wq = calculate_mmc_metrics(L, M, C)
        
        # convert wq from time units (min) to seconds for sl comparison [5]
        # Convert waiting time to seconds
        Wq_secs = Wq * 60 

        skill_names.append(skill)
        utilization_rates.append(rho)
        queue_lengths.append(Lq)
        wait_times_secs.append(Wq_secs)
        
        print(f"\n{skill} (λ={L}, μ={M}, c={C}):")
        print(f"  Utilization (ρ): {rho:.3f} (Target: 0.80) [10]")
        print(f"  Mean Queue Length (Lq): {Lq:.3f} [10]")
        print(f"  Utilization (ρ): {rho:.3f} (Target: 0.80)")
        print(f"  Mean Queue Length (Lq): {Lq:.3f}")
        print(f"  Avg Wait Time (Wq): {Wq:.3f} min ({Wq_secs:.2f} seconds)")

    # identify bottleneck (highest lq) [10]
    # Find the bottleneck (the one with the highest queue length)
    bottleneck_index = np.argmax(queue_lengths)
    bottleneck_skill = skill_names[bottleneck_index]
    print(f"\nBottleneck Identified (Highest Lq): {bottleneck_skill} [10]")
    print(f"\nBottleneck Identified (Highest Lq): {bottleneck_skill}")


    # --- visualization: dashboard (3 graphs) ---
    # --- Visualization: Dashboard (3 graphs) ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # graph 1: resource utilization
    # Graph 1: Resource Utilization
    bars = axs[0].bar(skill_names, utilization_rates, color='skyblue')
    axs[0].axhline(y=0.80, color='r', linestyle='--', label='Target Utilization (80%)')
    # highlight bottleneck
    # Highlight bottleneck
    bars[bottleneck_index].set_color('salmon') 
    axs[0].set_xlabel('Skill Group')
    axs[0].set_ylabel('Utilization (ρ)')
    axs[0].set_title('Agent Utilization vs. Target')
    axs[0].set_ylim(0, 1.1)
    axs[0].legend()
    axs[0].grid(axis='y', alpha=0.5)

    # graph 2: average queue length
    # Graph 2: Average Queue Length
    axs[1].bar(skill_names, queue_lengths, color='lightgreen')
    axs[1].set_title('Mean Queue Length (Lq)')
    axs[1].set_xlabel('Skill Group')
    axs[1].grid(axis='y', alpha=0.5)
    
    # graph 3: average wait time
    # Graph 3: Average Wait Time
    axs[2].bar(skill_names, wait_times_secs, color='orange')
    axs[2].axhline(y=20, color='r', linestyle='--', label='SL Target (20s)')
    axs[2].set_title('Avg Wait Time (Seconds)')
    axs[2].set_xlabel('Skill Group')
    axs[2].legend()
    axs[2].grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.show()

def analyze_scalability(skill_name, base_lambda, mu, c_initial):
    """
    Performs 'What-if analysis' for Scalability objective (25% surge) [10], [11].
    """
    # Check what happens if volume increases by 25%
    print("\n--- SCALABILITY ANALYSIS: 25% SURGE ---")
    
    surge_factor = 1.25
    surge_lambda = base_lambda * surge_factor # 25% surge in call volume [10]
    surge_lambda = base_lambda * surge_factor 
    
    # scenario 1: no change in agents (c_initial)
    # Scenario 1: Keep the same number of agents
    rho_initial, Lq_initial, Wq_initial = calculate_mmc_metrics(base_lambda, mu, c_initial)
    rho_surge_old_c, Lq_surge_old_c, Wq_surge_old_c = calculate_mmc_metrics(surge_lambda, mu, c_initial)
    
    print(f"\nSkill: {skill_name} (Initial Agents: {c_initial})")
    print(f"Initial Wq: {Wq_initial*60:.2f} seconds, Initial Utilization: {rho_initial:.2f}")

    print(f"\nScenario 1: Surge (λ={surge_lambda:.2f}) with NO Agent Increase (c={c_initial})")
    print(f"  New Utilization (ρ): {rho_surge_old_c:.3f}")
    print(f"  New Avg Wait Time (Wq): {Wq_surge_old_c*60:.2f} seconds")
    print(f"  Impact: Latency severely increases or system becomes unstable (Wq=inf) if ρ >= 1.")

    # scenario 2: find minimum new c required to maintain stability and performance
    # Scenario 2: Find how many new agents we need
    
    # we must determine the required agent count increase [10]. we test increasing 'c' 
    # until utilization (ρ) drops below the target utilization (e.g., 85% safety margin) 
    # or wq meets the sl target (e.g., wq < 20/60 min)
    
    c_required = c_initial
    Wq_target_min = 20 / 60  # target sl: 20 seconds [5]
    Wq_target_min = 20 / 60  # Target: 20 seconds
    
    # test incremental agents until the performance goal is met
    # Keep adding agents until we meet the target
    while True:
        rho_test, Lq_test, Wq_test = calculate_mmc_metrics(surge_lambda, mu, c_required)
        
        # check stability and performance against targets (e.g., target 85% sl in 20s)
        # Check if utilization is safe (< 95%) and wait time is good
        if rho_test < 0.95 and Wq_test < Wq_target_min:
             break
        c_required += 1
        
        # safety break to prevent infinite loop if parameters are impossible
        # Stop if we add too many agents (safety break)
        if c_required > c_initial + 20: 
             c_required = c_initial # revert to initial and break
             c_required = c_initial 
             break

    if c_required > c_initial:
        required_increase = c_required - c_initial
        print(f"\nScenario 2: Required Capacity Planning [11]")
        print(f"\nScenario 2: Required Capacity Planning")
        print(f"  New Required Agent Count (c) to handle surge: {c_required}")
        print(f"  Agent Increase Required: {required_increase} agents")
        print(f"  Predicted New Wq: {Wq_test*60:.2f} seconds, Predicted New Utilization: {rho_test:.2f}")
    elif rho_surge_old_c < 1:
        print("\nScenario 2: Current agent count is marginally sufficient to handle the 25% surge while maintaining stability.")
        print("\nScenario 2: Current agent count is enough for the surge.")

    # --- visualization 4: scalability analysis (sensitivity) ---
    # plot wq vs lambda to show the "hockey stick" curve
    lambda_values = np.linspace(base_lambda * 0.5, c_initial * mu * 0.99, 50) # up to 99% capacity
    # --- Visualization 4: Scalability Analysis (Sensitivity) ---
    # Plot Wq vs Lambda to show the "hockey stick" curve
    lambda_values = np.linspace(base_lambda * 0.5, c_initial * mu * 0.99, 50) 
    wq_values = []
    
    for l in lambda_values:
        _, _, w = calculate_mmc_metrics(l, mu, c_initial)
        wq_values.append(w * 60) # convert to seconds
        wq_values.append(w * 60) 

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, wq_values, label=f'Agents c={c_initial}', color='purple')
    plt.axvline(x=base_lambda, color='g', linestyle='--', label='Current Load')
    plt.axvline(x=surge_lambda, color='r', linestyle='--', label='Surge Load (+25%)')
    plt.title(f'Scalability Sensitivity: {skill_name} (Wait Time vs. Arrival Rate)')
    plt.xlabel('Arrival Rate (λ)')
    plt.ylabel('Avg Wait Time (Seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 4. test usage / main execution ---
# --- 4. Test Usage / Main Execution ---
if __name__ == "__main__":
    # run the performance analysis on the sample data
    # Run the analysis
    analyze_and_visualize_performance(SKILL_GROUP_DATA)

    # run a scalability test on the 'billing' group
    # we extract the parameters from the dictionary defined above
    # Test scalability for the Billing group
    skill_name = 'Billing'
    params = SKILL_GROUP_DATA[skill_name]
    
    analyze_scalability(skill_name, params['lambda'], params['mu'], params['c'])
