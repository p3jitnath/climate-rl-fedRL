program main

    use iso_c_binding
    use smartredis_client, only : client_type
    implicit none

#include "enum_fortran.inc"

    ! Define dimensions for the heating increment tensor (1D array of size dim1)
    integer, parameter :: dim1 = 1

    real(kind=c_double), dimension(dim1) :: py2f_redis
    real(kind=c_double), dimension(dim1) :: f2py_redis

    integer :: status
    logical :: compute_signal_found, start_signal_found
    type(client_type) :: client
    character(len=20) :: signal_key_compute, signal_key_start
    integer :: wait_time

    ! Variables for temperature calculation
    real(8) :: u, current_temperature, new_temperature
    real(8) :: initial_temperature

    ! Set the keys used to signal computation start and start
    signal_key_compute = "SIGCOMPUTE"
    signal_key_start = "SIGSTART"
    wait_time = 0.1  ! seconds to wait between checks

    ! Initialize the current temperature (300 - 273.15) / 100
    initial_temperature = (300.0d0 - 273.15d0) / 100.0d0
    current_temperature = initial_temperature

    ! Initialize the Redis client
    status = client%initialize(.false.)
    if (status .ne. SRNoError) error stop 'client%initialize failed'

    print *, "Waiting for computation or start signal..."

    ! Main loop to continuously wait for signals and perform actions
    do
       ! Check if the computation signal exists in Redis
       status = client%tensor_exists(signal_key_compute, compute_signal_found)

       ! Check if the start signal exists in Redis
       status = client%tensor_exists(signal_key_start, start_signal_found)

       ! If start signal is found, start the temperature to its initial value
       if (start_signal_found) then
          print *, "Start signal received. Resetting temperature..."
          f2py_redis(1) = initial_temperature  ! Store the start result
          current_temperature = initial_temperature  ! Reset the temperature

          print *, 'The value of f2py_redis is: ', f2py_redis

          ! Send the initial temperature to Redis under key "f2py_redis"
          status = client%put_tensor("f2py_redis", f2py_redis, shape(f2py_redis))
          if (status .ne. SRNoError) error stop 'client%put_tensor failed'

          print *, "Reset done. Result sent to Redis."

          ! Delete the start signal after processing it
          status = client%delete_tensor(signal_key_start)
          if (status .ne. SRNoError) error stop 'client%delete_tensor failed for SIGSTART'

          print *, "Temperature reset to initial value. Waiting for the next signal..."

       ! If computation signal is found, perform the computation
       else if (compute_signal_found) then
          print *, "Computation signal received. Starting computation..."

          ! Retrieve the heating increment (u) from Redis into py2f_redis
          status = client%unpack_tensor("py2f_redis", py2f_redis, shape(py2f_redis))
          if (status .ne. SRNoError) error stop 'client%unpack_tensor failed'

          ! Reset the  heating increment (u) after processing it
          status = client%delete_tensor("py2f_redis")
          if (status .ne. SRNoError) error stop 'client%delete_tensor failed for py2f_redis'

          ! Perform computation (update current temperature using forward subroutine)
          u = py2f_redis(1)  ! Heating increment received from Redis
          call forward(u, current_temperature, new_temperature)  ! Update temperature
          f2py_redis(1) = new_temperature  ! Store the computed result

          ! Update the current temperature for the next iteration
          current_temperature = new_temperature

          print *, 'The value of f2py_redis is: ', f2py_redis

          ! Send the updated temperature (new_temperature) to Redis under key "f2py_redis"
          status = client%put_tensor("f2py_redis", f2py_redis, shape(f2py_redis))
          if (status .ne. SRNoError) error stop 'client%put_tensor failed'

          print *, "Computation done. Result sent to Redis."

          ! Reset the computation signal after processing it
          status = client%delete_tensor(signal_key_compute)
          if (status .ne. SRNoError) error stop 'client%delete_tensor failed for SIGCOMPUTE'

          print *, "Computation signal start. Waiting for the next signal..."

       end if

       ! Wait for a bit before checking for signals again
       call sleep(wait_time)
    end do

end program main

! Subroutine to update temperature using heating increment and relaxation
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
