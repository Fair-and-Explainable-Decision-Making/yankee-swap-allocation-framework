from typing import List

from fair.constraint import LinearConstraint, PreferenceConstraint
from fair.feature import Course
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation, UniqueItemsValuation


def test_valid_constraint_valuation(
    bundle_250_301: list[ScheduleItem], all_courses_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([all_courses_constraint])

    assert valuation.independent(bundle_250_301) == 1
    assert valuation.value(bundle_250_301) == 2


def test_asymmetric_constraint_valuation(
    bundle_250_301: list[ScheduleItem],
    bundle_301_611: list[ScheduleItem],
    linear_constraint_250_301: LinearConstraint,
    all_courses_constraint: LinearConstraint,
):
    valuation = ConstraintSatifactionValuation([linear_constraint_250_301])
    assert valuation.value(bundle_250_301) == 1

    valuation = ConstraintSatifactionValuation([all_courses_constraint])
    assert valuation.value(bundle_301_611) == 2


def test_unique_item_adapter(
    schedule_item250: ScheduleItem, linear_constraint_250_301: LinearConstraint
):
    original_valuation = ConstraintSatifactionValuation([linear_constraint_250_301])
    unique_valuation = UniqueItemsValuation(original_valuation)
    bundle = [schedule_item250, schedule_item250]

    assert original_valuation.independent(bundle) == True
    assert original_valuation.value(bundle) == 2
    assert unique_valuation.independent(bundle) == True
    assert unique_valuation.value(bundle) == 1


def test_memoization(
    schedule_item250: ScheduleItem, all_items: List[ScheduleItem], course: Course
):
    constraint = PreferenceConstraint.from_item_lists(
        all_items, [[("250",), ("301",)]], [1], [course]
    )
    valuation = ConstraintSatifactionValuation([constraint])
    valuation.value([schedule_item250])
    valuation.value([schedule_item250])

    assert valuation._value_ct == 2
    assert valuation._unique_value_ct == 1

    constraint = PreferenceConstraint.from_item_lists(
        all_items, [[("250",), ("301",), ("611",)]], [2], [course]
    )
    valuation = ConstraintSatifactionValuation([constraint])
    valuation.value(all_items)
    before_value = valuation._unique_value_ct
    before_independent = valuation._unique_independent_ct
    valuation.value(all_items)

    assert valuation._unique_value_ct == before_value
    assert valuation._unique_independent_ct == before_independent


def test_disable_memoize(
    schedule_item250: ScheduleItem, all_items: List[ScheduleItem], course: Course
):
    constraint = PreferenceConstraint.from_item_lists(
        all_items, [[("250",), ("301",)]], [1], [course]
    )
    valuation = ConstraintSatifactionValuation([constraint], memoize=False)
    valuation.value([schedule_item250])
    valuation.value([schedule_item250])

    assert valuation._independent_ct == valuation._unique_independent_ct
    assert valuation._value_ct == valuation._unique_value_ct


def test_reset_memo_cache(
    schedule_item250: ScheduleItem, all_items: List[ScheduleItem], course: Course
):
    constraint = PreferenceConstraint.from_item_lists(
        all_items, [[("250",), ("301",)]], [1], [course]
    )
    valuation = ConstraintSatifactionValuation([constraint], memoize=False)
    valuation.value([schedule_item250])
    valuation.reset()

    assert valuation._independent_memo == {}
    assert valuation._value_memo == {}
    assert valuation._independent_ct == 0
    assert valuation._unique_independent_ct == 0
    assert valuation._value_ct == 0
    assert valuation._unique_value_ct == 0


def test_valuation_compilation(
    bundle_250_301: list[ScheduleItem], all_courses_constraint: LinearConstraint
):
    valuation = ConstraintSatifactionValuation([all_courses_constraint])
    compiled = valuation.compile()

    assert valuation.independent(bundle_250_301) == compiled.independent(bundle_250_301)
    assert valuation.value(bundle_250_301) == compiled.value(bundle_250_301)
