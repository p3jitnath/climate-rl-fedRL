subroutine climate_model(forcing, feedback_factor, heat_capacity, num_steps, final_temperature)
    implicit none
    real(8), intent(in) :: forcing, feedback_factor, heat_capacity
    integer, intent(in) :: num_steps
    real(8), intent(out) :: final_temperature

    real(8) :: delta_t, temperature, time_step, lambda, radiative_forcing
    integer :: i

    ! Initial conditions
    temperature = 15.0d0          ! Initial temperature in Celsius
    time_step = 1.0d0             ! Time step in years
    lambda = feedback_factor      ! Radiative feedback factor (W/m²/°C)

    ! Time loop to calculate temperature over time
    do i = 1, num_steps
       radiative_forcing = forcing - lambda * temperature
       delta_t = (radiative_forcing / heat_capacity) * time_step
       temperature = temperature + delta_t
    end do

    final_temperature = temperature
 end subroutine climate_model
