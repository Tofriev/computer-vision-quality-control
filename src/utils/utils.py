class Averager:
    """
    A utility class for calculating the running average, such as
    calculating average loss training loops.

    Attributes:
        current_total (float): The sum of all values added to the averager.
        iterations (float): The number of values added to the averager.

    Methods:
        send(value: float):
            Adds a new value to the averager and updates the running total and iteration count.

        value() -> float:
            Returns the current average of all values added to the averager. If no values
            have been added, returns 0 to prevent division by zero.

        reset():
            Resets the averager's state, clearing the current total and iteration count.
    """

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
