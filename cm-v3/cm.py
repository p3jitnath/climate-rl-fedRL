import numpy as np
import scm  # Import the compiled Fortran module from f2py


def forward(forcing, feedback_factor, heat_capacity, temperature_t):
    """
    Call the Fortran climate model to compute temperature for one time step.

    Parameters
    ----------
    forcing : float
        External forcing value (e.g., radiative forcing due to CO2).
    feedback_factor : float
        Feedback factor controlling the radiative feedback in the system.
    heat_capacity : float
        Effective heat capacity of the Earth's system (J/m²/°C).
    temperature_t : float
        Current temperature at time t.

    Returns
    -------
    temperature_t1 : float
        Updated temperature for the next time step (t+1).
    """

    # Call the Fortran climate model function
    return scm.forward(forcing, feedback_factor, heat_capacity, temperature_t)
