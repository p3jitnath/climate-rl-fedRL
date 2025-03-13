import numpy as np
import scm  # Import the compiled Fortran module
from scipy.optimize import minimize


def fortran_climate_model(forcing, heat_capacity, feedback_factors, num_steps):
    """
    Calls the Fortran subroutine to simulate a biased climate model with time-dependent feedback factors.

    Parameters
    ----------
    forcing : float
        External forcing (e.g., radiative forcing in W/m²).
    heat_capacity : float
        Effective heat capacity of the Earth's system (J/m²/°C).
    feedback_factors : array_like
        An array of time-dependent feedback factors, one per timestep.
    num_steps : int
        Number of time steps for the simulation.

    Returns
    -------
    float
        Final temperature after simulation.
    """
    return scm.biased_climate_model(
        forcing, heat_capacity, feedback_factors, num_steps
    )


def objective_function(
    feedback_factors, forcing, heat_capacity, observed_temp, num_steps=100
):
    """
    Objective function for optimization, computes the squared error between the model output and the observed temperature.

    Parameters
    ----------
    feedback_factors : array_like
        An array of time-dependent feedback factors (one per timestep).
    forcing : float
        External forcing (e.g., radiative forcing in W/m²).
    heat_capacity : float
        Effective heat capacity of the Earth's system (J/m²/°C).
    observed_temp : float
        Observed temperature (target value) to minimize the difference with.
    num_steps : int, optional
        Number of time steps in the model simulation (default is 100).

    Returns
    -------
    float
        The squared error between the final model temperature and the observed temperature.
    """
    final_temp = fortran_climate_model(
        forcing, heat_capacity, feedback_factors, num_steps
    )
    return (final_temp - observed_temp) ** 2


# Observed temperature (target value)
observed_temp = 25.0  # Example observed temperature at the final timestep

# Number of timesteps
num_steps = 100

# Initial guesses for time-dependent feedback factors (one feedback factor per timestep)
initial_guess_feedback = np.full(
    num_steps, 0.5
)  # Start with an initial guess of 0.5 for all timesteps

# Fixed forcing and heat capacity (we will optimize feedback factors only)
forcing = 3.7  # Example external forcing
heat_capacity = 1000.0  # Example heat capacity

# Optimize the time-dependent feedback factors
result = minimize(
    objective_function,
    initial_guess_feedback,
    args=(forcing, heat_capacity, observed_temp, num_steps),
    method="L-BFGS-B",
)

# Extract the optimized time-dependent feedback factors
feedback_factors_opt = result.x

# Display the optimized feedback factors
print("Optimized Feedback Factors:", feedback_factors_opt)

# Resultant temperature with optimized feedback factors
final_temperature = fortran_climate_model(
    forcing, heat_capacity, feedback_factors_opt, num_steps
)
print(f"Final Temperature with Optimized Feedback: {final_temperature:.4f} °C")
