subroutine biased_climate_model(forcing, heat_capacity, feedback_factors, num_steps, final_temperature)
    implicit none
    real(8), intent(in) :: forcing, heat_capacity
    real(8), intent(in) :: feedback_factors(num_steps)  ! Time-dependent feedback factors
    integer, intent(in) :: num_steps
    real(8), intent(out) :: final_temperature

    real(8) :: delta_t, temperature, time_step, radiative_forcing
    integer :: i
    real(8) :: bias_factor

    ! Initial conditions
    temperature = 15.0d0          ! Initial temperature in Celsius
    time_step = 1.0d0             ! Time step in years

    ! Time loop to calculate temperature over time
    do i = 1, num_steps
        ! Introduce a bias factor at each timestep
        bias_factor = 1.0d0 + 0.1d0 * sin(0.01d0 * real(i) * temperature)

        ! Use the time-dependent feedback factor
        radiative_forcing = forcing - feedback_factors(i) * temperature * bias_factor

        ! Calculate temperature change
        delta_t = (radiative_forcing / heat_capacity) * time_step

        ! Update the temperature
        temperature = temperature + delta_t
    end do

    ! Output the final temperature
    final_temperature = temperature
end subroutine biased_climate_model
