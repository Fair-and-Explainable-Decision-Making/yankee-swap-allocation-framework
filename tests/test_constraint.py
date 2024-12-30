from typing import List

import numpy as np
import scipy

from fair.constraint import (
    CourseTimeConstraint,
    MutualExclusivityConstraint,
    PreferenceConstraint,
    indicator,
)
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem


def test_indicator(bundle_250_301: list[ScheduleItem]):
    # sparse
    ind = indicator(bundle_250_301, 3, True)
    np.testing.assert_array_equal(ind.toarray().flatten(), [1, 0, 1])

    # dense
    ind = indicator(bundle_250_301, 3, False)
    np.testing.assert_array_equal(ind.flatten(), [1, 0, 1])


def test_preference_with_multiple_features(
    course: Course,
    section: Section,
    bundle_250_301: list[ScheduleItem],
    bundle_250_301_3: list[ScheduleItem],
    schedule: list[ScheduleItem],
):
    constraint = PreferenceConstraint.from_item_lists(
        schedule,
        [[("250", 2), ("301", 1), ("611", 1)]],
        [1],
        [course, section],
        False,
    )

    assert constraint.satisfies(bundle_250_301)
    assert not constraint.satisfies(bundle_250_301_3)


def test_linear_constraint(
    course: Course,
    bundle_250_301_2: list[ScheduleItem],
    schedule: list[ScheduleItem],
):
    # sparse
    constraint = PreferenceConstraint.from_item_lists(
        schedule,
        [[("250",), ("301",), ("611",)]],
        [1],
        [course],
        True,
    )

    assert not constraint.satisfies(bundle_250_301_2)

    # dense
    constraint = PreferenceConstraint.from_item_lists(
        schedule,
        [[("250",), ("301",), ("611",)]],
        [1],
        [course],
        False,
    )

    assert not constraint.satisfies(bundle_250_301_2)


def test_linear_constraint_250_301(
    linear_constraint_250_301: PreferenceConstraint,
    schedule_item250: ScheduleItem,
    bundle_250_301: list[ScheduleItem],
):
    assert not linear_constraint_250_301.satisfies(bundle_250_301)
    assert linear_constraint_250_301.satisfies([schedule_item250])


def test_time_constraint(
    all_items: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    slot: Slot,
    weekday: Weekday,
):
    constraint = CourseTimeConstraint.from_items(all_items, slot, weekday)

    assert not constraint.satisfies(bundle_250_301)
    assert constraint.satisfies(bundle_301_611)


def test_section_constraint(
    items_repeat_section: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_250_250_2: list[ScheduleItem],
    course: Course,
    section: Section,
):
    constraint = MutualExclusivityConstraint.from_items(items_repeat_section, course)

    assert constraint.satisfies(bundle_250_301)
    assert not constraint.satisfies(bundle_250_250_2)


def test_constrained_items(
    all_items: list[ScheduleItem],
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    course_time_constraint: CourseTimeConstraint,
):
    assert schedule_item250 in course_time_constraint.constrained_items(all_items)

    ct_active = course_time_constraint.constrained_items(all_items)
    assert (
        len(set(ct_active[schedule_item250]).intersection(ct_active[schedule_item301]))
        > 0
    )


def test_sparse_addition(course: Course, schedule: List[ScheduleItem]):
    constraint1 = PreferenceConstraint.from_item_lists(
        schedule, [[("250",), ("301",)]], [1], [course], True
    )
    constraint2 = PreferenceConstraint.from_item_lists(
        schedule, [[("301",), ("611",)]], [1], [course], True
    )
    constraint = constraint1 + constraint2

    assert constraint.A.shape[0] == constraint1.A.shape[0] + constraint2.A.shape[0]
    assert constraint._sparse


def test_dense_addition(course: Course, schedule: List[ScheduleItem]):
    constraint1 = PreferenceConstraint.from_item_lists(
        schedule, [[("250",), ("301",)]], [1], [course], False
    )
    constraint2 = PreferenceConstraint.from_item_lists(
        schedule, [[("301",), ("611",)]], [1], [course], False
    )
    constraint = constraint1 + constraint2

    assert constraint.A.shape[0] == constraint1.A.shape[0] + constraint2.A.shape[0]
    assert not constraint._sparse


def test_to_desnse_to_sparse(course: Course, schedule: List[ScheduleItem]):
    constraint = PreferenceConstraint.from_item_lists(
        schedule, [[("250",), ("301",)]], [1], [course], False
    )

    constraint = constraint.to_dense()
    assert not scipy.sparse.issparse(constraint.A)
    assert not scipy.sparse.issparse(constraint.b)

    constraint = constraint.to_sparse()
    assert scipy.sparse.issparse(constraint.A)
    assert scipy.sparse.issparse(constraint.b)


def test_prune(
    all_items: list[ScheduleItem],
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    slot: Slot,
    weekday: Weekday,
):
    constraint = CourseTimeConstraint.from_items(all_items, slot, weekday)
    constraint_pruned = constraint.prune()

    assert constraint_pruned.A.shape[0] == constraint_pruned.b.shape[0]
    assert constraint.A.shape[0] > constraint_pruned.A.shape[0]

    assert constraint.satisfies(bundle_250_301) == constraint_pruned.satisfies(
        bundle_250_301
    )
    assert constraint.satisfies(bundle_301_611) == constraint_pruned.satisfies(
        bundle_301_611
    )
