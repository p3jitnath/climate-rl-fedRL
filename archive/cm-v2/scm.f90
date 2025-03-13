subroutine forward(forcing, feedback_factor, heat_capacity, temperature_t, temperature_t1)
    implicit none
    real(8), intent(in) :: forcing, feedback_factor, heat_capacity, temperature_t
    real(8), intent(out) :: temperature_t1

    real(8) :: delta_t, radiative_forcing

    ! Compute radiative forcing based on the current temperature
    radiative_forcing = forcing - feedback_factor * temperature_t

    ! Calculate the temperature change based on the heat capacity and radiative forcing
    delta_t = (radiative_forcing / heat_capacity)

    ! Update the temperature for the next time step
    temperature_t1 = temperature_t + delta_t

end subroutine forward
