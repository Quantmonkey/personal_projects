import numpy as np
cimport numpy as np
cimport cython

from datetime import datetime

from utils import logger_utils
from utils.constants import *

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeCircularBuffer:
    cdef:
        double[:] data
        double[:] timestamps
        double max_seconds_time_length
        size_t head
        size_t tail
        size_t size
        int max_size
#
    def __cinit__(self, 
                  max_seconds_time_length, 
                  max_size):
        self.max_seconds_time_length = max_seconds_time_length
        self.max_size = max_size
        self.data = np.zeros(max_size, dtype=float)
        self.timestamps = np.zeros(max_size, dtype=float)
        self.head = 0
        self.tail = 0
        self.size = 0
#
    def append(self, 
               double data_point, 
               double timestamp):
        """Adds a new data point to the circular buffer, where it will
        delete old/stale data and update the pointers.

        Args:
            double data_point
            double timestamp
        """
        
        # remove oldest data points that are outside the max time length
        while (
            (self.size > 0) 
            and (
                timestamp 
                - self.timestamps[self.tail] 
                > self.max_seconds_time_length
            )
        ):
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
        
        # add new data point
        self.data[self.head] = data_point
        self.timestamps[self.head] = timestamp
        self.head = (self.head + 1) % self.max_size
        self.size += 1
#
    def get_earliest_data(self):
        return self.data[self.tail]
#
    def get_earliest_timestamp(self):
        return self.timestamps[self.tail]
#
    def get_latest_data(self):
        return self.data[self.head-1]
#
    def get_latest_timestamp(self):
        return self.timestamps[self.head-1]
#
    def get_data(self):
        if self.head > self.tail:
            return self.data[self.tail:self.head]
        else:
            return np.concatenate(
                (self.data[self.tail:], 
                self.data[:self.head])
            )
#
    def get_timestamps(self):
        if self.head > self.tail:
            return self.timestamps[self.tail:self.head]
        else:
            return np.concatenate(
                (self.timestamps[self.tail:], 
                self.timestamps[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    def get_time_size(self):
        return (
            self.get_latest_timestamp()
            -self.get_earliest_timestamp()
        )
#
    cpdef set_max_seconds_time_length(self, 
                                      double current_timestamp, 
                                      double new_max_seconds_time_length):
        """Change the max seconds time length, and removes any data points
        that are older than the current timestamp as provided.

        Args:
            double current_timestamp Provided current timestamp
            double new_max_seconds_time_length: New time buffer length
        """

        self.max_seconds_time_length = new_max_seconds_time_length

        while (
            self.size > 0 
            and (
                (current_timestamp - self.timestamps[self.tail]) 
                > self.max_seconds_time_length
            )
        ):
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
#
    cpdef reset_data(self):
        with nogil:
            self.data[:] = 0
            self.timestamps[:] = 0
            self.head = 0
            self.tail = 0
            self.size = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeCircularBufferRollingMean:
    cdef:
        double[:] data
        double[:] timestamps
        double max_seconds_time_length
        double sum
        int max_size
        int size
        int head
        int tail
#
    def __init__(self, 
                 max_seconds_time_length, 
                 max_size):
        self.max_seconds_time_length = max_seconds_time_length
        self.max_size = max_size
        self.data = np.zeros(max_size, dtype=float)
        self.timestamps = np.zeros(max_size, dtype=float)
        self.sum = 0
        self.size = 0
        self.head = 0
        self.tail = 0
#    
    cpdef update(self, 
                 double data_point, 
                 double timestamp):
        """Updates the internal states required to get the rolling
        mean of an array without requiring array calculations
        Removes any data that is older than the rolling window allowed.
        The principal assumption of this data structure is that max_size
        is materially larger (in time) than the max_seconds_time_length.

        Args:
            double data_point: The data point we are adding
            double timestamp: The timestamp of the data point
        """
        # remove oldest data points that are outside the max time length
        while (
            (
                self.size > 0 
                and (
                    (timestamp - self.timestamps[self.tail]) 
                    > self.max_seconds_time_length
                )
            )
        ):
            self.sum -= self.data[self.tail]
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
        
        # add new data point
        self.data[self.head] = data_point
        self.timestamps[self.head] = timestamp
        self.sum += data_point
        self.head = (self.head + 1) % self.max_size
        self.size += 1
#
    cpdef double get_sum(self):
        return self.sum
#
    cpdef double get_mean(self):
        """Gets the rolling mean after updating
        Does not protect from self.size == 0
        """
        return self.sum / self.size
#
    cpdef set_max_seconds_time_length(self, 
                                      double current_timestamp, 
                                      double new_max_seconds_time_length):
        """Change the max seconds time length, and removes any data points
        that are older than the current timestamp as provided.

        Args:
            double current_timestamp Provided current timestamp
            double new_max_seconds_time_length: New time buffer length
        """

        if not np.isclose(
            self.max_seconds_time_length, 
            new_max_seconds_time_length, 
            atol=ATOL_IN_SECONDS_BEFORE_RESIZING_TIME_CIRCULAR_BUFFER
        ):
            self.max_seconds_time_length = new_max_seconds_time_length

            while (
                self.size > 0 
                and (
                    (current_timestamp - self.timestamps[self.tail]) 
                    > self.max_seconds_time_length
                )
            ):
                self.sum -= self.data[self.tail]
                self.tail = (self.tail + 1) % self.max_size
                self.size -= 1
#
    def get_earliest_data(self):
        return self.data[self.tail]
#
    def get_earliest_timestamp(self):
        return self.timestamps[self.tail]
#
    def get_latest_data(self):
        return self.data[self.head-1]
#
    def get_latest_timestamp(self):
        return self.timestamps[self.head-1]
#
    def get_data(self):
        if self.head > self.tail:
            return self.data[self.tail:self.head]
        else:
            return np.concatenate(
                (self.data[self.tail:], 
                self.data[:self.head])
            )
#
    def get_timestamps(self):
        if self.head > self.tail:
            return self.timestamps[self.tail:self.head]
        else:
            return np.concatenate(
                (self.timestamps[self.tail:], 
                self.timestamps[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    def get_time_size(self):
        return (
            self.get_latest_timestamp()
            -self.get_earliest_timestamp()
        )
#
    cpdef reset_data(self):
        with nogil:
            self.data[:] = 0
            self.timestamps[:] = 0
            self.head = 0
            self.tail = 0
            self.size = 0
            self.sum = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeCircularBufferRollingVar:
    cdef:
        double[:] data
        double[:] timestamps
        double max_seconds_time_length
        int max_size
        int size
        double mean
        double sum
        double sum_of_squared
        int head
        int tail
        double tail_data
#
    def __init__(self, max_size, max_seconds_time_length):
        self.max_size = max_size
        self.max_seconds_time_length = max_seconds_time_length
        self.data = np.zeros(max_size, dtype=float)
        self.timestamps = np.zeros(max_size, dtype=float)
        self.size = 0
        self.sum = 0
        self.sum_of_squared = 0
        self.head = 0
        self.tail = 0
#   
    cpdef update(self, double data_point, double timestamp):
        """Updates the internal states required to get the rolling
        variance of an array without requiring array calculations
        Removes any data that is older than the rolling window allowed.
        The principal assumption of this data structure is that max_size
        is materially larger (in time) than the max_seconds_time_length.

        Args:
            double data_point: The data point we are adding
            double timestamp: The timestamp of the data point
        """

        cdef:
            double tail_data

        while (
            self.size > 0 
            and (
                (timestamp - self.timestamps[self.tail]) 
                > self.max_seconds_time_length
            )
        ):
            tail_data = self.data[self.tail]
            self.sum -= tail_data
            self.sum_of_squared -= tail_data**2
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
        
        # add new data point
        self.data[self.head] = data_point
        self.timestamps[self.head] = timestamp
        self.sum += data_point
        self.sum_of_squared += data_point**2
        self.head = (self.head + 1) % self.max_size
        self.size += 1
#
    cpdef double get_sum(self):
        return self.sum
#
    cpdef double get_mean(self):
        """Gets the rolling mean after updating
        Does not protect from self.size == 0
        """
        return self.sum / self.size
#
    cpdef double get_var(self):
        """Does not protect against self.size < 2
        """

        mean = self.get_mean()
        var = (self.sum_of_squared/(self.size)) - (mean**2)

        return var
#
    cpdef double get_std(self):

        cdef:
            double var

        var = self.get_var()
        if var <= 0:
            return 0
        return np.sqrt(var)
#
    cpdef set_max_seconds_time_length(self, 
                                      double current_timestamp, 
                                      double new_max_seconds_time_length):
        """Change the max seconds time length, and removes any data points
        that are older than the current timestamp as provided.

        Args:
            double current_timestamp Provided current timestamp
            double new_max_seconds_time_length: New time buffer length
        """

        if not np.isclose(
            self.max_seconds_time_length, 
            new_max_seconds_time_length, 
            atol=ATOL_IN_SECONDS_BEFORE_RESIZING_TIME_CIRCULAR_BUFFER
        ):
            self.max_seconds_time_length = new_max_seconds_time_length

            while (
                self.size > 0 
                and (
                    (current_timestamp - self.timestamps[self.tail]) 
                    > self.max_seconds_time_length
                )
            ):
                tail_data = self.data[self.tail]
                self.sum -= tail_data
                self.sum_of_squared -= tail_data**2
                self.tail = (self.tail + 1) % self.max_size
                self.size -= 1
#
    def get_earliest_data(self):
        return self.data[self.tail]
#
    def get_earliest_timestamp(self):
        return self.timestamps[self.tail]
#
    def get_latest_data(self):
        return self.data[self.head-1]
#
    def get_latest_timestamp(self):
        return self.timestamps[self.head-1]
#
    def get_data(self):
        if self.head > self.tail:
            return self.data[self.tail:self.head]
        else:
            return np.concatenate(
                (self.data[self.tail:], 
                self.data[:self.head])
            )
#
    def get_timestamps(self):
        if self.head > self.tail:
            return self.timestamps[self.tail:self.head]
        else:
            return np.concatenate(
                (self.timestamps[self.tail:], 
                self.timestamps[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    def get_time_size(self):
        return (
            self.get_latest_timestamp()
            -self.get_earliest_timestamp()
        )
#
    cpdef reset_data(self):
        with nogil:
            self.data[:] = 0
            self.timestamps[:] = 0
            self.size = 0
            self.sum = 0
            self.sum_of_squared = 0
            self.head = 0
            self.tail = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeCircularBufferRollingLinearRegression:
    cdef:
        double[:] data_x
        double[:] data_y
        double[:] timestamps
        double max_seconds_time_length
        int max_size
        int size
        int head
        int tail

        double sum_x
        double sum_y
        double sum_xy
        double sum_x_squared
        double sum_y_squared
#
    def __init__(self, max_size, max_seconds_time_length):
        self.max_size = max_size
        self.max_seconds_time_length = max_seconds_time_length

        self.size = 0
        self.head = 0
        self.tail = 0

        self.data_x = np.zeros(max_size, dtype=float)
        self.data_y = np.zeros(max_size, dtype=float)
        self.timestamps = np.zeros(max_size, dtype=float)

        self.sum_x = 0
        self.sum_y = 0
        self.sum_xy = 0
        self.sum_x_squared = 0
        self.sum_y_squared = 0
#   
    cpdef update(self, 
                 double x_data_point, 
                 double y_data_point, 
                 double timestamp):
        """Updates the internal states required to get the rolling
        coefficient of two time series without requiring array calculations
        Removes any data that is older than the rolling window allowed.
        The principal assumption of this data structure is that max_size
        is materially larger (in time) than the max_seconds_time_length.

        Args:
            double data_point: The data point we are adding
            double timestamp: The timestamp of the data point
        """

        cdef:
            double x_tail_data

        while (
            self.size > 0 
            and (
                (timestamp - self.timestamps[self.tail]) 
                > self.max_seconds_time_length
            )
        ):
            x_tail_data = self.data_x[self.tail]
            y_tail_data = self.data_y[self.tail]
            self.sum_x -= x_tail_data
            self.sum_y -= y_tail_data
            self.sum_xy -= x_tail_data*y_tail_data
            self.sum_x_squared -= x_tail_data**2
            self.sum_y_squared -= y_tail_data**2
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
        
        # add new data point
        self.data_x[self.head] = x_data_point
        self.data_y[self.head] = y_data_point
        self.timestamps[self.head] = timestamp
        self.sum_x += x_data_point
        self.sum_y += y_data_point
        self.sum_xy += x_data_point*y_data_point
        self.sum_x_squared += x_data_point**2
        self.sum_y_squared += y_data_point**2
        self.head = (self.head + 1) % self.max_size
        self.size += 1
#
    cpdef double get_rolling_coefficient(self):
        """Gets the rolling coefficient after updating
        Does not protect from self.size < 2
        """
        cdef:
            double numerator
            double denominator
            double b1

        numerator = (self.size*self.sum_xy) - (self.sum_x*self.sum_y)
        denominator = (self.size*self.sum_x_squared) - self.sum_x**2

        if denominator != 0:
            b1 = numerator / denominator
        else:
            return 0

        return b1
#
    cpdef set_max_seconds_time_length(self, 
                                      double current_timestamp, 
                                      double new_max_seconds_time_length):
        """Change the max seconds time length, and removes any data points
        that are older than the current timestamp as provided.

        Args:
            double current_timestamp Provided current timestamp
            double new_max_seconds_time_length: New time buffer length
        """

        if not np.isclose(
            self.max_seconds_time_length, 
            new_max_seconds_time_length, 
            atol=ATOL_IN_SECONDS_BEFORE_RESIZING_TIME_CIRCULAR_BUFFER
        ):
            self.max_seconds_time_length = new_max_seconds_time_length

            while (
                self.size > 0 
                and (
                    (current_timestamp - self.timestamps[self.tail]) 
                    > self.max_seconds_time_length
                )
            ):
                x_tail_data = self.data_x[self.tail]
                y_tail_data = self.data_y[self.tail]
                self.sum_x -= x_tail_data
                self.sum_y -= y_tail_data
                self.sum_xy -= x_tail_data*y_tail_data
                self.sum_x_squared -= x_tail_data**2
                self.sum_y_squared -= y_tail_data**2
                self.tail = (self.tail + 1) % self.max_size
                self.size -= 1
#
    def get_earliest_X_y_data(self):
        return self.data_x[self.tail], self.data_y[self.tail]
#
    def get_earliest_timestamp(self):
        return self.timestamps[self.tail]
#
    def get_oldest_X_y_data(self):
        return self.data_x[self.head-1], self.data_y[self.head-1]
#
    def get_latest_timestamp(self):
        return self.timestamps[self.head-1]
#
    def get_X_y_data(self):
        if self.head > self.tail:
            return (
                self.data_x[self.tail:self.head],
                self.data_y[self.tail:self.head]
            )
        else:
            return (
                np.concatenate(
                    (self.data_x[self.tail:], 
                    self.data_x[:self.head])
                ),
                np.concatenate(
                    (self.data_y[self.tail:], 
                    self.data_y[:self.head])
                ),
            )
#
    def get_timestamps(self):
        if self.head > self.tail:
            return self.timestamps[self.tail:self.head]
        else:
            return np.concatenate(
                (self.timestamps[self.tail:], 
                self.timestamps[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    def get_time_size(self):
        return (
            self.get_latest_timestamp()
            -self.get_earliest_timestamp()
        )
#
    cpdef reset_data(self):
        with nogil:
            self.size = 0
            self.head = 0
            self.tail = 0

            self.data_x[:] = 0
            self.data_y[:] = 0
            self.timestamps[:] = 0

            self.sum_x = 0
            self.sum_y = 0
            self.sum_xy = 0
            self.sum_x_squared = 0
            self.sum_y_squared = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CircularBufferRollingMean:
    cdef:
        double[:] data
        int max_size
        int size
        double sum
        size_t head
        size_t tail
#        
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = np.zeros(max_size, dtype=float)
        self.size = 0
        self.sum = 0
        self.head = 0
        self.tail = 0
#    
    def update(self, data_point):
        """Updates the internal states required to get the rolling
        mean of an array without requiring array calculations
        Removes older data if the buffer is full

        Args:
            double data_point: The data point we are adding
        """

        if self.size < self.max_size:
            self.size += 1
        else:
            self.sum -= self.data[self.tail]
            self.tail = (self.tail + 1) % self.max_size

        self.data[self.head] = data_point
        self.sum += data_point
        self.head = (self.head + 1) % self.max_size
#
    cpdef double get_sum(self):
        return self.sum
#    
    cpdef double get_mean(self):
        """Gets the rolling mean after updating
        Does not protect from self.size == 0
        """
        return self.sum / self.size
#
    def get_earliest_data(self):
        return self.data[self.tail]
#
    def get_latest_data(self):
        return self.data[self.head-1]
#
    def get_data(self):
        if self.head > self.tail:
            return self.data[self.tail:self.head]
        else:
            return np.concatenate(
                (self.data[self.tail:], 
                self.data[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    cpdef reset_data(self):
        with nogil:
            self.data[:] = 0
            self.size = 0
            self.sum = 0
            self.head = 0
            self.tail = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CircularBufferRollingVar:
    cdef:
        double[:] data
        int max_size
        int size
        double sum
        double sum_of_squared
        int head
        int tail
#  
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = np.zeros(max_size, dtype=float)
        self.size = 0
        self.sum = 0
        self.sum_of_squared = 0
        self.head = 0
        self.tail = 0
#   
    def update(self, double data_point):
        """Updates the internal states required to get the rolling
        variance of an array without requiring array calculations
        Removes older data if the buffer is full

        Args:
            double data_point: The data point we are adding
        """

        cdef:
            double tail_data

        if self.size < self.max_size:
            self.size += 1
        else:
            tail_data = self.data[self.tail]
            self.sum -= tail_data
            self.sum_of_squared -= tail_data**2
            self.tail = (self.tail + 1) % self.max_size
        
        self.data[self.head] = data_point
        self.sum += data_point
        self.sum_of_squared += data_point**2
        self.head = (self.head + 1) % self.max_size
#
    cpdef double get_sum(self):
        return self.sum
#
    cpdef double get_mean(self):
        """Gets the rolling mean after updating
        Does not protect from self.size == 0
        """
        return self.sum / self.size
#
    cpdef double get_var(self):
        """Does not protect against self.size < 2
        """
        mean = self.get_mean()
        var = (self.sum_of_squared/(self.size)) - (mean**2)

        return var
#
    cpdef double get_std(self):
        """Does not protect against self.size < 2
        """

        var = self.get_var()
        if var <= 0:
            return 0
        return np.sqrt(var)
#
    def get_earliest_data(self):
        return self.data[self.tail]
#
    def get_latest_data(self):
        return self.data[self.head-1]
#
    def get_data(self):
        if self.head > self.tail:
            return self.data[self.tail:self.head]
        else:
            return np.concatenate(
                (self.data[self.tail:], 
                self.data[:self.head])
            )
#
    def get_size(self):
        return self.size
#
    cpdef reset_data(self):
        with nogil:
            self.data[:] = 0
            self.size = 0
            self.sum = 0
            self.sum_of_squared = 0
            self.head = 0
            self.tail = 0
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class VectorRollingMean:
    cdef:
        list data
        int size
        double sum
#        
    def __init__(self):
        self.data = []
        self.size = 0
        self.sum = 0
#    
    def update(self, double data_point):
        """Updates the internal states required to get the rolling
        mean of a vector without requiring array calculations

        Args:
            double data_point: The data point we are adding
        """

        self.data.append(data_point)
        self.size += 1
        self.sum += data_point
#
    def get_sum(self):
        return self.sum
#
    def get_mean(self):
        """Gets the rolling mean after updating
        Does not protect from self.size == 0
        """
        return self.sum/self.size
#
    def get_earliest_data(self):
        return self.data[0]
#
    def get_latest_data(self):
        return self.data[self.size-1]
#
    def get_data(self):
        return self.data
#
    def get_size(self):
        return self.size
#
    def reset_data(self):
        self.data = []
        self.size = 0
        self.sum = 0

