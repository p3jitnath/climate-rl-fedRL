import numpy as np
import scm  # Import the compiled Fortran climate model module
from scipy.optimize import minimize


# Wrapper function to call the Fortran subroutine for climate model simulation
def fortran_climate_model(forcing, feedback_factor, heat_capacity, num_steps):
    """
    Calls the Fortran climate model subroutine and returns the final temperature.

    Parameters:
    forcing (float): External forcing (e.g., CO2 radiative forcing in W/m²)
    feedback_factor (float): Climate feedback parameter (W/m²/°C)
    heat_capacity (float): Effective heat capacity of Earth's system (J/m²/°C)
    num_steps (int): Number of time steps in the model simulation

    Returns:
    float: Final temperature after simulation
    """
    return scm.climate_model(
        forcing, feedback_factor, heat_capacity, num_steps
    )


# Objective function for optimization
def objective_function(params, observed_temp, num_steps=100):
    """
    Computes the squared error between the climate model output and the observed temperature.

    Parameters:
    params (list): [forcing, feedback_factor, heat_capacity]
    observed_temp (float): Observed temperature (target value)
    num_steps (int): Number of time steps in the model simulation

    Returns:
    float: Squared error between model output and observed temperature
    """
    forcing, feedback_factor, heat_capacity = params
    final_temp = fortran_climate_model(
        forcing, feedback_factor, heat_capacity, num_steps
    )
    return (final_temp - observed_temp) ** 2  # Return squared error


# Observed temperature (target) for comparison
observed_temp = 25.0  # Replace with actual observed temperature data

# Initial guesses for [forcing, feedback_factor, heat_capacity]
params_t0 = [3.7, 0.5, 1000.0]

# Perform optimization to minimize the objective function
result = minimize(
    objective_function, params_t0, args=(observed_temp,), method="Nelder-Mead"
)

# Extract optimized parameters
forcing_opt, feedback_factor_opt, heat_capacity_opt = result.x

# Display the results
print(f"Optimized Forcing: {forcing_opt:.4f} W/m²")
print(f"Optimized Feedback Factor: {feedback_factor_opt:.4f} W/m²/°C")
print(f"Optimized Heat Capacity: {heat_capacity_opt:.4f} J/m²/°C")

# Compute the resultant temperature with the optimized parameters
final_temp_opt = fortran_climate_model(
    forcing_opt, feedback_factor_opt, heat_capacity_opt, num_steps=100
)
print(f"Resultant Temperature: {final_temp_opt:.4f} °C")
