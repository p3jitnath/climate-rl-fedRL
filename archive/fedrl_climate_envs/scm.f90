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
    logical :: compute_signal_found, start_signal_found, compute_data_found
    type(client_type) :: client
    character(len=20) :: k_sigcompute, k_sigstart, k_f2py, k_py2f, cmd_cid
    integer :: wait_time, cid

    ! Variables for temperature calculation
    real(8) :: u, current_temperature, new_temperature
    real(8) :: initial_temperature

    ! Command-line argument handling for cid
    if (command_argument_count() < 1) then
        print *, "Error: Please provide a cid as a commandline argument."
        stop
    end if

    call get_command_argument(1, cmd_cid)
    read(cmd_cid, *) cid
    if (cid < 0) then
        print *, "Error: Seed must be a non-negative integer."
        stop
    end if

    ! Set the keys used to signal computation start and start with the cid
    write(k_sigcompute, '("SIGCOMPUTE_S", i0)') cid
    write(k_sigstart, '("SIGSTART_S", i0)') cid
    write(k_f2py,'("f2py_redis_s", i0)') cid
    write(k_py2f,'("py2f_redis_s", i0)') cid

    wait_time = 0.0001 ! seconds to wait between checks

    ! Initialize the current temperature (300 - 273.15) / 100
    ! Supports 8 different cids 0-7 (inclusive)
    initial_temperature = (300.0d0 + 10*(cid - 1) - 273.15d0) / 100.0d0
    current_temperature = initial_temperature

    ! Initialize the Redis client
    status = client%initialize(.false.)
    if (status .ne. SRNoError) error stop 'client%initialize failed'

    print *, "Waiting for computation or start signal..."

    ! Main loop to continuously wait for signals and perform actions
    do
        ! Check if the computation signal exists in Redis
        status = client%tensor_exists(k_sigcompute, compute_signal_found)

        ! Check if the start signal exists in Redis
        status = client%tensor_exists(k_sigstart, start_signal_found)

        ! Check if the compute data exists in Redis
        status = client%tensor_exists(k_py2f, compute_data_found)

        ! If start signal is found, start the temperature to its initial value
        if (start_signal_found) then
            print *, "Start signal received. Resetting temperature..."

            ! Delete the start signal before processing it
            status = client%delete_tensor(k_sigstart)
            if (status .ne. SRNoError) error stop 'client%delete_tensor failed for SIGSTART'
            call sleep(wait_time)

            f2py_redis(1) = initial_temperature ! Store the start result
            current_temperature = initial_temperature ! Reset the temperature

            print *, 'The value of f2py_redis is: ', f2py_redis

            ! Send the initial temperature to Redis under key "f2py_redis"
            status = client%put_tensor(k_f2py, f2py_redis, shape(f2py_redis))
            if (status .ne. SRNoError) error stop 'client%put_tensor failed'

            print *, "Reset done. Result sent to Redis."
            print *, "Temperature reset to initial value. Waiting for the next signal..."

        ! If computation signal is found, perform the computation
        else if (compute_signal_found .and. compute_data_found) then
            print *, "Computation signal and data received. Starting computation..."

            ! Reset the computation signal before processing it
            call sleep(wait_time)
            status = client%delete_tensor(k_sigcompute)
            if (status .ne. SRNoError) error stop 'client%delete_tensor failed for SIGCOMPUTE'

            ! Retrieve the heating increment (u) from Redis into "py2f_redis"
            status = client%unpack_tensor(k_py2f, py2f_redis, shape(py2f_redis))
            if (status .ne. SRNoError) error stop 'client%unpack_tensor failed'

            ! Reset the heating increment (u) after processing it
            call sleep(wait_time)
            status = client%delete_tensor(k_py2f)
            if (status .ne. SRNoError) error stop 'client%delete_tensor failed for py2f_redis'

            ! Perform computation (update current temperature using forward subroutine)
            u = py2f_redis(1) ! Heating increment received from Redis
            call forward(u, current_temperature, new_temperature) ! Update temperature
            f2py_redis(1) = new_temperature ! Store the computed result

            ! Update the current temperature for the next iteration
            current_temperature = new_temperature

            print *, 'The value of f2py_redis is: ', f2py_redis

            ! Send the updated temperature (new_temperature) to Redis under key "f2py_redis"
            status = client%put_tensor(k_f2py, f2py_redis, shape(f2py_redis))
            if (status .ne. SRNoError) error stop 'client%put_tensor failed'

            print *, "Computation done. Result sent to Redis."
            print *, "Computation signal reset. Waiting for the next signal..."

        end if

        ! Wait for a bit before checking for signals again
        call sleep(wait_time)
    end do

    end program main

    ! Subroutine to update temperature using heating increment and relaxation
    subroutine forward(u, current_temperature, new_temperature)
    ! Input and output variables
    real(8), intent(in) :: u ! Heating increment
    real(8), intent(in) :: current_temperature ! Current temperature
    real(8), intent(out) :: new_temperature ! Updated temperature

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
