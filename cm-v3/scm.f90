subroutine forward(u, current_temperature, new_temperature)
    ! Input and output variables
    real(8), intent(in) :: u                    ! Heating increment
    real(8), intent(in) :: current_temperature  ! Current temperature
    real(8), intent(out) :: new_temperature     ! Updated temperature

    real(8) :: observed_temperature, physics_temperature, division_constant
    real(8) :: relaxation

    ! Define the observed and physics temperatures (constants)
    observed_temperature = (321.75d0 - 273.15d0) / 100.0d0
    physics_temperature = (380.0d0 - 273.15d0) / 100.0d0
    division_constant = physics_temperature - observed_temperature

    ! Update temperature based on the heating increment and relaxation term
    new_temperature = current_temperature + u
    relaxation = (physics_temperature - current_temperature) * 0.2d0 / division_constant
    new_temperature = new_temperature + relaxation

end subroutine forward
