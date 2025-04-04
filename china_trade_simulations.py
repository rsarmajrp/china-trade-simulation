import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import streamlit as st

st.title("China Trade - Payoff Simulations")

def simulate_stock_returns(S0=100, target_mean_return=0.08, target_stdev=0.20, T=1, dt=1/252, num_simulations=10000):
    """
    Simulates stock returns using a log-normal process with calibrated parameters.
    
    Parameters:
    S0 (float): Initial stock price
    target_mean_return (float): Target mean return (e.g., 0.08 for 8%)
    target_stdev (float): Target standard deviation
    T (float): Time horizon in years (default is 1 year)
    dt (float): Time step size (default is daily, 1/252)
    num_simulations (int): Number of Monte Carlo simulations
    
    Returns:
    tuple: Simulated mean return, median return, standard deviation, min, max, and return distribution
    """
    # Compute sigma using the closed-form solution
    target_variance = target_stdev ** 2
    sigma = np.sqrt(np.log(1 + target_variance / (1 + target_mean_return) ** 2))
    
    # Compute mu using the closed-form solution, adjusted for compounding
    mu = np.log(1 + target_mean_return)
    
    # Initialize final stock prices
    np.random.seed(42)
    final_prices = np.full(num_simulations, S0, dtype=np.float64)
    
    # Vectorized simulation
    N = int(T / dt)  # Number of time steps
    dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, N))
    exp_term = (mu - 0.5 * sigma**2) * dt + sigma * dW
    final_prices *= np.exp(np.sum(exp_term, axis=1))
    
    # Compute simulated returns
    simulated_returns = (final_prices / S0) - 1
    
    return simulated_returns

def payoff(simulated_returns, intial_repo, put_premium, put_strike):
    delta1_leg_payoff = simulated_returns + intial_repo
    put_payoff = np.maximum(put_strike - (1+ simulated_returns), 0) - put_premium
    total_payoff = delta1_leg_payoff + put_payoff
    return total_payoff
    
target_mean_return_input = st.sidebar.slider("% Expected return", min_value=0.0, max_value=20.0, value=8.0, step=0.5, key="target_mean_return_input")
st.sidebar.caption('The log-normal process is designed to generate the mean annualized geometric return')
target_stdev_input = st.sidebar.slider("% Volatility", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key="target_stdev_input")
st.sidebar.caption('The annualized standard deviation along a simulation path')

# Parameters
S0 = 100  # Initial stock price
target_mean_return = target_mean_return_input / 100  # Expected drift (8%)
target_stdev = target_stdev_input / 100  # Volatility (40%)
T = 1  # Time horizon (10 years)
num_simulations = 100000  # Number of Monte Carlo simulations
dt = 1/252 # length of time steps (daily frequency)

# Run simulation
simulated_returns = simulate_stock_returns(S0, target_mean_return, target_stdev, T, dt, num_simulations)

# Calculate payoffs for each simulation
intial_repo =.118
put_premium = .1196
put_strike = 0.972
simulated_payoffs = payoff(simulated_returns, intial_repo, put_premium, put_strike)


# create a dataframe from returns and simulated_payoffs
df = pd.DataFrame({'Index Returns': simulated_returns, 'Trade Payoffs': simulated_payoffs})
df_summary = df.describe()#.to_clipboard()
# Select and rearrage the rows
df2 = df_summary.loc[['mean', 'std', '25%', '50%', '75%']]
# Display results
st.dataframe(
    df2.style.format({
        df2.columns[0]: "{:.1%}",
        df2.columns[1]: "{:.1%}"
    })
) 

# Plot histogram
plt.hist((1+simulated_returns)**(1/T)-1, bins=100, edgecolor='black', alpha=0.5, density = True, histtype='stepfilled')
plt.axvline(x=target_mean_return, color='grey', linestyle='dashed', label='Target Return')
plt.gca().xaxis.set_major_formatter(PercentFormatter(1)) 
plt.xlabel("Terminal Return (Ann)")
plt.ylabel("Density (%)")
plt.title("Log-normal Monte Carlo Simulation of Index Returns")
#indicate mu and sigma as text on the plot on the top left corner
#plt.text(50, num_simulations/20, , fontsize=12)

plt.text(0.95, 0.95, f'Exp Return = {target_mean_return*100:.1f}% \n Std Dev = {target_stdev*100:.1f}% \n T = {T} year', 
         horizontalalignment='right', 
         verticalalignment='top', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.5))

st.pyplot(plt.gcf())