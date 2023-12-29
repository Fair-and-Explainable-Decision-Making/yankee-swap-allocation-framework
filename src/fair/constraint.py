from collections import defaultdict
from typing import Any, List, Union

import numpy as np
import scipy
from scipy.sparse import dok_array

from .feature import BaseFeature, Course, Section, Slot
from .item import BaseItem, ScheduleItem


def indicator(bundle: List[BaseItem], extent: int, sparse: bool):
    """Indicator vector for bundle over domain of features

    This method returns a vector of length extent with a 1 at the index of each item
    in the bundle

    Args:
        bundle (List[BaseItem]): Items whose index we would like to identify
        extent (int): Maximum index in domain
        sparse (bool): Should the vector returned be sparse

    Returns:
        scipy.sparse.dok_array: Indicator vector of bundle indices
    """
    rows = extent
    if sparse:
        ind = dok_array((rows, 1), dtype=np.int_)
    else:
        ind = np.zeros((rows, 1), dtype=np.int_)

    for item in bundle:
        ind[item.index, 0] = True

    if sparse:
        ind = ind.tocsr()

    return ind


class BaseConstraint:
    pass


class LinearConstraint(BaseConstraint):
    """Constraints that can be expressed in the form A*x <= b"""

    def __init__(
        self,
        A: Union[dok_array, np.array],
        b: Union[dok_array, np.array],
        extent: int,
    ):
        """
        Args:
            A (Union[dok_array, np.array]): Constraint matrix
            b (Union[dok_array, np.array]): Row capacities
            extent (int): Largest possible index value
        """
        if type(A) != type(b):
            raise TypeError(f"type of A: {type(A)} and b: {type(b)} must be identical")

        if scipy.sparse.issparse(A):
            self.A = A.tocsr()
            self.b = b.tocsr()
            self._sparse = True
        else:
            self.A = A
            self.b = b
            self._sparse = False

        self.extent = extent

    def to_sparse(self):
        if self._sparse:
            return self

        return LinearConstraint(
            scipy.sparse.csr_matrix(self.A),
            scipy.sparse.csr_matrix(self.b),
            self.extent,
        )

    def to_dense(self):
        if not self._sparse:
            return self

        return LinearConstraint(self.A.to_dense(), self.b.to_dense(), self.extent)

    def satisfies(self, bundle: List[BaseItem]):
        """Determine if bundle satisfies this constraint

        Args:
            bundle (List[BaseItem]): Items in the bundle

        Returns:
            bool: True if the constraint is satisfied; False otherwise
        """
        ind = indicator(bundle, self.extent, self._sparse)
        product = self.A @ ind

        # apparently <= is much less efficient than using < and != separately
        if self._sparse:
            less_than = (product < self.b).toarray().flatten()
            equal_to = ~(product != self.b).toarray().flatten()

            return np.prod([lt or eq for lt, eq in zip(less_than, equal_to)])
        else:
            return np.prod(product <= self.b)

    def constrained_items(self, items: BaseItem):
        """Determine if, and for what constraint, each item is constrained

        Args:
            items (BaseItem): Items to evaluate

        Returns:
            Dict(BaseItem, List[int]): List of constraints (rows of A) where each item is constrained
        """
        active_map = defaultdict(list)
        for i in range(self.A.shape[0]):
            for item in items:
                if self.A[i, item.index] != 0:
                    active_map[item].append(i)

        return active_map

    def __add__(self, other):
        """Adds A and b (row-wise) between self and other

        Args:
            other (LinearConstraint): LinearConstraint to be added

        Raises:
            TypeError: Type of self and other must match
            TypeError: Sparsity must match between self and other
            ValueError: Column numbers must match between self and other

        Returns:
            LinearConstraint: sum of self and other
        """
        if type(other) != type(self):
            raise TypeError(f"type of other: {type(other)} must match {type(self)}")

        if other._sparse != self._sparse:
            raise TypeError(
                f"sparsity of other: {other._sparse} must match {self._sparse}"
            )

        if other.A.shape[1] != self.A.shape[1]:
            raise ValueError(
                f"column dimension for other: {other.A.shape[1]} must match {self.A.shape[1]}"
            )

        extent = max(self.extent, other.extent)
        if self._sparse:
            A = scipy.sparse.vstack([self.A, other.A])
            b = scipy.sparse.vstack([self.b, other.b])
        else:
            A = np.vstack([self.A, other.A])
            b = np.vstack([self.b, other.b])

        return LinearConstraint(A, b, extent)


class PreferenceConstraint(LinearConstraint):
    @staticmethod
    def from_item_lists(
        schedule: List[BaseItem],
        preferred_values: List[List[Any]],
        limits: List[int],
        preferred_feature: BaseFeature,
        sparse: bool = False,
    ):
        """A helper method for constructing preference constraints

        This constraint ensures that the bundle contains a limited number of pre-selected items from
        each provided category (e.g. Physics, Chemistry, etc. for course items).

        Args:
            schedule: (List[BaseItem], optional): Universe of all items under consideration
            preferred_values (List[List[Any]]): Each list is a category and values in that list are preferred
            limits (List[int]): The maximum number of items desired per category
            preferred_feature (BaseFeature): The feaure in terms of which preferred values are expressed
            sparse (bool): Should A and b be sparse matrices. Defaults to False.

        Raises:
            IndexError: Number of categories must match among preferred_items and limits

        Returns:
            PreferrenceConstraint: A: (categories x features domain), b: (categories x 1)
        """
        if len(preferred_values) != len(limits):
            raise IndexError("item and limit lists must have the same length")

        rows = len(preferred_values)
        cols = max([item.index for item in schedule]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i in range(rows):
            for value in preferred_values[i]:
                for item in schedule:
                    if item.value(preferred_feature) == value:
                        A[
                            i,
                            item.index,
                        ] = 1
            b[i, 0] = limits[i]

        if not sparse:
            A = A.todense()
            b = b.todense()

        return LinearConstraint(A, b, cols)


class CourseTimeConstraint(LinearConstraint):
    @staticmethod
    def from_items(
        items: List[ScheduleItem],
        slot: Slot,
        sparse: bool = False,
    ):
        """Helper method for creating constraints that prevent course time overlap

        A bundle satisfies this constraint only if no two courses meet at the same time.

        Args:
            items (List[ScheduleItem]): Possibly time-conflicting items
            slot (Slot): Feature for time slots
            sparse (bool): Should A and b be sparse matrices. Defaults to False.

        Returns:
            CourseTimeConstraint: A: (time slots x features domain), b: (time slots x 1)
        """
        rows = len(slot.times)
        cols = max([item.index for item in items]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, tm in enumerate(slot.times):
            items_at_time = [item for item in items if tm in item.value(slot)]
            for item in items_at_time:
                A[i, item.index] = 1
            b[i, 0] = 1

        if not sparse:
            A = A.todense()
            b = b.todense()

        return LinearConstraint(A, b, cols)


class MutualExclusivityConstraint(LinearConstraint):
    @staticmethod
    def from_items(
        items: List[ScheduleItem],
        exclusive_feature: BaseFeature,
        sparse: bool = False,
    ):
        """Helper method for creating constraints that prevent scheduling multiple sections of the same class

        Args:
            items (List[ScheduleItem]): Items, possibly having same value for exclusive_feature
            exclusive_feature (BaseFeature): Feature that must remain exclusive
            sparse (bool, optional): Should A and b be sparse matrices. Defaults to False.

        Returns:
            MutualExclusivityConstraint: A: (exclusive_feature domain x features domain), b: (exclusive_feature domain x 1)
        """
        rows = len(exclusive_feature.domain)
        cols = max([item.index for item in items]) + 1
        A = dok_array((rows, cols), dtype=np.int_)
        b = dok_array((rows, 1), dtype=np.int_)

        for i, excl in enumerate(exclusive_feature.domain):
            items_for_course = [
                item for item in items if item.value(exclusive_feature) == excl
            ]
            for item in items_for_course:
                A[i, item.index] = 1
            b[i, 0] = 1

        if not sparse:
            A = A.todense()
            b = b.todense()

        return LinearConstraint(A, b, cols)
