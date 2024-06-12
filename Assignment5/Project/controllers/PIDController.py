class PIDController:
    """
    A Proportional-Integral-Derivative (PID) Controller class.

    Args:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        output_max (float): Maximum output value.
        output_min (float): Minimum output value.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        output_max (float): Maximum output value.
        output_min (float): Minimum output value.
        prev_error (float): Previous error value.
        accumulative_error (float): Accumulated error value.

    Methods:
        get_control_command: Calculates the control command based on the current error.

    """

    def __init__(self, Kp:float, Ki:float, Kd:float, output_min:float, output_max:float):
        """
        Initialize the PIDController with the specified gains and output limits.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            output_max (float): Maximum output value.
            output_min (float): Minimum output value.
        """
        # Constants
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Limits
        self.output_max = output_max
        self.output_min = output_min

        # relevant errors
        self.prev_error = 0.
        self.accumulative_error = 0.
    
    def get_control_command(self, current_error: float, dt:float)-> float:
        """
        Calculate the control command based on the current error and time difference.

        Args:
            current_error (float): The current error value.
            dt (float): The time difference since the last control command.

        Returns:
            float: The calculated control command.
        """
        # compute control signal
        integral = self.accumulative_error + current_error * dt
        diff = (current_error - self.prev_error) / dt
        control_command = self.Kp * current_error + self.Ki * integral + self.Kd * diff

        # update errors
        self.prev_error = current_error
        self.accumulative_error = integral

        return max(self.output_min, min(control_command, self.output_max))