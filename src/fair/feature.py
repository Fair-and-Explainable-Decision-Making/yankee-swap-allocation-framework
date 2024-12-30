import datetime
import itertools
from typing import Any, List, Tuple

import pandas as pd


class DomainError(Exception):
    pass


class FeatureError(Exception):
    pass


class BaseFeature:
    """A named, typed, and ordered space"""

    def __init__(self, name: str, domain: list):
        """
        Args:
            name (str): Feature name
            domain (list): Exhaustive, ordered list of possible feature values
        """
        self.name = name
        self.domain = domain

    def index(self, value: Any):
        return self.domain.index(value)

    def __repr__(self):
        return f"{self.name}: [{self.domain[0]} ... {self.domain[-1]}]"

    def __hash__(self):
        return hash(self.name) ^ hash(tuple(self.domain))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Course(BaseFeature):
    """Ordered space of courses"""

    def __init__(self, domain):
        """
        Args:
            domain (_type_): Exhaustive, ordered list of possible feature values
        """
        super().__init__("course", domain)


def parse_time_range(time_range: str):
    """Converts a time range string into two datetime objects

    Args:
        time_range (str): A string of the form "%H:%M - %H:%M"

    Returns:
        List[pd.Timestamp]: Lower and upper timestamps in range
    """
    left, right = time_range.split("-")

    return [pd.to_datetime(left.strip()).time(), pd.to_datetime(right.strip()).time()]


def slot_list(frequency: str):
    """List of time slots at given frequency

    Args:
        frequency (str): A valid time series offset: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

    Returns:
        List[datetime.Time]: List of times deonting time slots
    """
    return [
        dt.time()
        for dt in pd.date_range("1900/01/01 00:00", "1900/01/02 00:00", freq=frequency)
    ]


def slots_for_time_range(time_range: str, time_slots: List[datetime.time]):
    """Convert string to list of time slots

    Include all time slots contained in the range.

    Args:
        time_range (str): A string with format "%H:%M - %H:%M"
        time_slots (List[datetime.time]): A list of all time slots

    Returns:
        Tuple[datetime.Time]: All slots in the time range
    """
    rng = parse_time_range(time_range)
    values = []
    for tm in time_slots:
        if tm >= rng[0] and tm < rng[1]:
            values.append(tm)

    return tuple(values)


class Slot(BaseFeature):
    """Ordered space of time slots"""

    @staticmethod
    def from_time_ranges(time_ranges: List[str], frequency: str):
        """Helper method for creating a slot feature from a list of time ranges

        The domain of the Slot includes a tuple for each time range, which contains all
        slots in the range.

        Args:
            time_ranges (List[str]): Time ranges from which to generate domain
            frequency (str): A valid time series offset: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

        Returns:
            Slot: Slot constructed from time ranges
        """
        time_slots = slot_list(frequency)
        domain = []
        for rng in time_ranges:
            item_slots = slots_for_time_range(rng, time_slots)
            domain.append(item_slots)

        return Slot(time_slots, domain)

    def __init__(self, times: List[datetime.time], domain: List[Tuple[datetime.time]]):
        """
        Args:
            times (List[datetime.time]): All valid time slots
            domain (List[Tuple[datetime.time]]): Exhaustive, ordered list of all possible time slots ranges
        """
        super().__init__("slot", domain)
        self.times = times


class Weekday(BaseFeature):
    """Day of the week"""

    def __init__(self):
        self.days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

        domain = [
            comb
            for i in range(1, len(self.days) + 1)
            for comb in itertools.combinations(self.days, i)
        ]

        super().__init__("weekday", domain)


class Section(BaseFeature):
    """Ordered space of course sections"""

    def __init__(self, domain):
        """
        Args:
            domain (_type_): Exhaustive, ordered list of possible feature values
        """
        super().__init__("section", domain)
