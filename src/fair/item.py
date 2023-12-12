from typing import Any, List

import numpy as np
import pandas as pd

from .feature import (
    BaseFeature,
    Course,
    DomainError,
    FeatureError,
    Section,
    Slot,
    slot_list,
    slots_for_time_range,
)


class BaseItem:
    """Item defined over multiple features"""

    def __init__(
        self,
        name: str,
        features: List[BaseFeature],
        values: List[Any],
        capacity: int = 1,
    ):
        """
        Args:
            features (List[BaseFeature]): Features revelvant for this item
            values (List[Any]): Value of each feature from its domain
            capacity (int): Number of times item can be allocated

        Raises:
            FeatureError: Values and features must correspond 1:1
            DomainError: Features can only take values from their domain
        """
        self.name = name
        self.features = features
        self.values = values
        self.capacity = capacity

        # validate cardinality
        if len(self.values) != len(self.features):
            raise FeatureError("values must correspond to features 1:1")

        # validate domain
        for feature, value in zip(self.features, self.values):
            if value not in feature.domain:
                raise DomainError(f"invalid value for feature '{feature}'")

    def value(self, feature: BaseFeature):
        """Value associated with a given feature

        Args:
            feature (BaseFeature): Feature for which value is required

        Raises:
            FeatureError: Feature must have been registered during inititialization

        Returns:
            Any: Value for feature
        """
        try:
            return self.values[self.features.index(feature)]
        except IndexError:
            raise FeatureError("feature unknown for this item")

    def index(self, features: List[BaseFeature] = None):
        """Position of item in canonical order

        The domains of features provided as input are ordered according to their cartesian product.
        This method maps the feature values of the present item the associated point in that product.

        Args:
            features (List[BaseFeature], optional): Subset of features from initialization. Defaults to None.

        Raises:
            FeatureError: Features provided must be a subset of those provided during initialization

        Returns:
            Any: Point associated with item in the cartesian product of feature domains
        """
        features = self.features if features is None else features
        mult = 1
        idx = 0
        for feature in features:
            if feature not in self.features:
                raise FeatureError(f"feature {feature} not valid for {self}")
            idx += feature.index(self.value(feature)) * mult
            mult *= len(feature.domain)

        return idx

    def __repr__(self):
        return f"{self.name}: {[self.value(feature) for feature in self.features]}"

    def __hash__(self):
        return hash(self.name) ^ hash(
            tuple([self.value(feature) for feature in self.features])
        )

    def __lt__(self, other):
        return self.__hash__() < hash(other)

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class ScheduleItem(BaseItem):
    """An item representing a class in a schedule"""

    @staticmethod
    def parse_excel(path: str, frequency: str = "15T"):
        """Read and parse schedule items from excel file

        Args:
            path (str): Full path to excel file

        Returns:
            List[ScheduleItem]: All items that could be extracted from excel file
        """
        with open(path, "rb") as fd:
            df = pd.read_excel(fd)
        df = df[
            df.columns.intersection(
                ["Catalog", "Section", "Mtg Time", "CICScapacity", "Categories"]
            )
        ].dropna()

        course = Course(df["Catalog"].unique())
        section = Section(df["Section"].unique())
        time_slots = slot_list(frequency)
        slot = Slot.from_time_ranges(df["Mtg Time"].unique(), "15T")
        features = [course, section, slot]
        items = []
        for _, row in df.iterrows():
            values = [
                row["Catalog"],
                row["Section"],
                slots_for_time_range(row["Mtg Time"], time_slots),
            ]
            try:
                items.append(
                    ScheduleItem(
                        features,
                        values,
                        int(row["CICScapacity"]),
                        row["Categories"] if "Categories" in df else None,
                    )
                )
            except DomainError:
                pass

        return items

    def __init__(
        self,
        features: List[BaseFeature],
        values: List[Any],
        capacity: int = 1,
        category: str = None,
    ):
        """An Item appropriate for course scheduling

        Args:
            features (List[BaseFeature]): Features revelvant for this item
            values (List[Any]): Value of each feature from its domain
            capacity (int): Number of times item can be allocated. Defaults to 1.
            category (str, optional): Topic for course. Defaults to None.
        """
        super().__init__("schedule", features, values, capacity)
        self.category = category