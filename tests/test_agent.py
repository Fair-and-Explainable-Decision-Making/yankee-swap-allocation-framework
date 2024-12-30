from fair.agent import (
    LegacyStudent,
    Student,
    exchange_contribution,
    marginal_contribution,
)
from fair.constraint import (
    CourseTimeConstraint,
    MutualExclusivityConstraint,
    PreferenceConstraint,
)
from fair.feature import BaseFeature, Course, Slot, Weekday
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation, StudentValuation


def test_exchange_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert (
        exchange_contribution(
            course_valuation, [all_items[0]], all_items[0], all_items[1]
        )
        == True
    )

    assert (
        exchange_contribution(
            course_valuation,
            all_items,
            all_items[2],
            all_items[1],
        )
        == False
    )


def test_marginal_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert marginal_contribution(course_valuation, [all_items[0]], all_items[1]) == 1
    assert (
        marginal_contribution(
            course_valuation, [all_items[0], all_items[1]], all_items[2]
        )
        == 0
    )


def test_student(
    course: Course,
    slot: Slot,
    weekday: Weekday,
    schedule: list[ScheduleItem],
):
    preferred_constr = PreferenceConstraint.from_item_lists(
        schedule, [[("250",), ("301",), ("611",)]], [2], [course]
    )
    course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday)
    course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course)
    student = Student(
        StudentValuation([preferred_constr, course_time_constr, course_sect_constr])
    )

    # two non-conflicting courses
    assert student.value([schedule[0], schedule[2]]) == 2
    # two courses at the same time
    assert student.value([schedule[1], schedule[2]]) == 1
    # two sections of the same course
    assert student.value([schedule[0], schedule[1]]) == 1


def test_get_desired_item_indexes(
    schedule: list[ScheduleItem],
    course: Course,
):
    item1 = schedule[0]
    item2 = schedule[2]
    actual_desired = [item1, item2]
    preferred_constr = PreferenceConstraint.from_item_lists(
        schedule,
        [[(item1.value(course),), (item2.value(course),)]],
        [2],
        [course],
    )
    student = Student(StudentValuation([preferred_constr]))
    leg_student = LegacyStudent(student, actual_desired, course)
    actual_desired_idxs = [item.index for item in actual_desired]
    computed_desired = leg_student.get_desired_items_indexes(schedule)

    assert actual_desired_idxs == computed_desired
