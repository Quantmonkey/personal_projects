import pickle
import numpy as np
import pandas as pd

from ..utils import *

class CalendarFeature:
    def __init__(self,
                 data_reference,
                 calendar_feature):
        self.data_reference = data_reference
        self.calendar_feature = calendar_feature

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        if self.data_reference == 'prediction_date':
            reference_date = prediction_date
        elif self.data_reference == 'realization_date':
            reference_date = realization_date

        if self.calendar_feature == 'day_of_week':
            calendar_feature = reference_date.day_of_week
        elif self.calendar_feature == 'week':
            calendar_feature = reference_date.week
        elif self.calendar_feature == 'month':
            calendar_feature = reference_date.month
        elif self.calendar_feature == 'day_of_year':
            calendar_feature = reference_date.day_of_year
        elif self.calendar_feature == 'days_in_month':
            calendar_feature = reference_date.days_in_month

        # elif self.calendar_feature == 'is_leap_year':
        #     calendar_feature = reference_date.is_leap_year
        # elif self.calendar_feature == 'is_month_start':
        #     calendar_feature = reference_date.is_month_start
        # elif self.calendar_feature == 'is_month_end':
        #     calendar_feature = reference_date.is_month_end
        # elif self.calendar_feature == 'is_quarter_start':
        #     calendar_feature = reference_date.is_quarter_start
        # elif self.calendar_feature == 'is_quarter_end':
        #     calendar_feature = reference_date.is_quarter_end
        # elif self.calendar_feature == 'is_year_start':
        #     calendar_feature = reference_date.is_year_start
        # elif self.calendar_feature == 'is_year_end':
        #     calendar_feature = reference_date.is_year_end

        return calendar_feature
