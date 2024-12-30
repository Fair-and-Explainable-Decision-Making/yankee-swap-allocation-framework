import os
from typing import List

import numpy as np
import pytest

from fair.constraint import (
    CourseTimeConstraint,
    LinearConstraint,
    MutualExclusivityConstraint,
    PreferenceConstraint,
)
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair.valuation import ConstraintSatifactionValuation


@pytest.fixture
def course_domain():
    return ["250", "301", "611"]


@pytest.fixture
def course(course_domain: list[int]):
    return Course(course_domain)


@pytest.fixture
def slot():
    return Slot([1, 2, 3, 4, 5, 6, 7], [(1, 2), (2, 3), (4, 5), (6, 7)])


@pytest.fixture
def weekday():
    return Weekday()


@pytest.fixture
def section():
    return Section([1, 2, 3])


@pytest.fixture
def features(course: Course, slot: Slot, section: Section):
    return [course, slot, section]


@pytest.fixture
def schedule_item250(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return ScheduleItem(
        [course, slot, weekday, section], ["250", (1, 2), ("Mon",), 1], 0
    )


@pytest.fixture
def schedule_item250_2(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return ScheduleItem(
        [course, slot, weekday, section], ["250", (4, 5), ("Mon",), 2], 1
    )


@pytest.fixture
def schedule_item301(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return ScheduleItem(
        [course, slot, weekday, section], ["301", (2, 3), ("Mon",), 1], 2
    )


@pytest.fixture
def schedule_item301_2(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return ScheduleItem(
        [course, slot, weekday, section], ["301", (4, 5), ("Mon",), 1], 3
    )


@pytest.fixture
def schedule_item611(course: Course, slot: Slot, weekday: Weekday, section: Section):
    return ScheduleItem(
        [course, slot, weekday, section], ["611", (4, 5), ("Mon",), 1], 4
    )


@pytest.fixture
def bundle_250_301(schedule_item250: ScheduleItem, schedule_item301: ScheduleItem):
    return [schedule_item250, schedule_item301]


@pytest.fixture
def bundle_250_301_2(schedule_item250: ScheduleItem, schedule_item301_2: ScheduleItem):
    return [schedule_item250, schedule_item301_2]


@pytest.fixture
def bundle_250_301_3(schedule_item250_2: ScheduleItem, schedule_item301: ScheduleItem):
    return [schedule_item250_2, schedule_item301]


@pytest.fixture
def bundle_250_250_2(schedule_item250: ScheduleItem, schedule_item250_2: ScheduleItem):
    return [schedule_item250, schedule_item250_2]


@pytest.fixture
def bundle_301_611(schedule_item301: ScheduleItem, schedule_item611: ScheduleItem):
    return [schedule_item301, schedule_item611]


@pytest.fixture
def all_courses_constraint(course: Course, all_items: list[ScheduleItem]):
    return PreferenceConstraint.from_item_lists(
        all_items, [[("250",), ("301",), ("611",)]], [2], [course]
    )


@pytest.fixture
def linear_constraint_250_301(course: Course, bundle_250_301: list[ScheduleItem]):
    return PreferenceConstraint.from_item_lists(
        bundle_250_301, [[("250",), ("301",)]], [1], [course]
    )


@pytest.fixture
def course_time_constraint(
    all_items: list[ScheduleItem],
    slot: Slot,
    weekday: Weekday,
):
    return CourseTimeConstraint.from_items(all_items, slot, weekday)


@pytest.fixture
def all_items(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]


@pytest.fixture
def items_repeat_section(
    schedule_item250: ScheduleItem,
    schedule_item250_2: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item250_2, schedule_item301, schedule_item611]


@pytest.fixture
def course_valuation(all_courses_constraint: PreferenceConstraint):
    return ConstraintSatifactionValuation([all_courses_constraint])


@pytest.fixture
def schedule(course: Course, slot: Slot, weekday: Weekday, section: Section):
    features = [course, slot, weekday, section]
    items = [
        ScheduleItem(features, ["250", (1, 2), ("Mon",), 1], 0),
        ScheduleItem(features, ["250", (4, 5), ("Mon",), 2], 1),
        ScheduleItem(features, ["301", (4, 5), ("Mon",), 1], 2),
        ScheduleItem(features, ["301", (6, 7), ("Mon",), 2], 3),
        ScheduleItem(features, ["611", (6, 7), ("Mon",), 1], 4),
    ]
    return items


@pytest.fixture
def simple_schedule(
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250, schedule_item301, schedule_item611]


@pytest.fixture
def simple_schedule2(
    schedule_item250_2: ScheduleItem,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return [schedule_item250_2, schedule_item301, schedule_item611]


@pytest.fixture
def excel_schedule_path():
    return os.path.join(
        os.path.dirname(__file__), "../resources/fall2023schedule-2.xlsx"
    )


@pytest.fixture
def excel_schedule_path_with_cats():
    return os.path.join(
        os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
    )


@pytest.fixture
def global_constraints(
    schedule: list[ScheduleItem],
    course: Course,
    section: Section,
    slot: Slot,
    weekday: Weekday,
):
    course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday)
    course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course)

    return [course_time_constr, course_sect_constr]


@pytest.fixture
def simple_global_constraints(
    simple_schedule: list[ScheduleItem],
    course: Course,
    slot: Slot,
    weekday: Weekday,
):
    course_time_constr = CourseTimeConstraint.from_items(simple_schedule, slot, weekday)
    course_sect_constr = MutualExclusivityConstraint.from_items(simple_schedule, course)

    return [course_time_constr, course_sect_constr]


@pytest.fixture
def renaissance1(
    schedule: List[ScheduleItem],
    global_constraints: List[LinearConstraint],
    course: Course,
    section: Section,
):
    return RenaissanceMan(
        [[schedule[0], schedule[2]], [schedule[4]]],
        [1, 1],
        1,
        2,
        course,
        section,
        global_constraints,
        schedule,
        seed=0,
    )


@pytest.fixture
def renaissance2(
    schedule: List[ScheduleItem],
    global_constraints: List[LinearConstraint],
    course: Course,
    section: Section,
):
    return RenaissanceMan(
        [[schedule[0], schedule[2]], [schedule[4]]],
        [1, 1],
        1,
        2,
        course,
        section,
        global_constraints,
        schedule,
        seed=1,
    )


@pytest.fixture
def renaissance3(
    schedule: List[ScheduleItem],
    global_constraints: List[LinearConstraint],
    course: Course,
    section: Section,
):
    return RenaissanceMan(
        [[schedule[0], schedule[2]], [schedule[4]]],
        [1, 1],
        2,
        3,
        course,
        section,
        global_constraints,
        schedule,
        seed=0,
    )


@pytest.fixture
def student(
    simple_schedule: list[ScheduleItem],
    simple_global_constraints: list[LinearConstraint],
    course: Course,
    section: Section,
    schedule_item250: ScheduleItem,
    schedule_item301: ScheduleItem,
):
    return RenaissanceMan(
        [[schedule_item250, schedule_item301]],
        [1],
        1,
        1,
        course,
        section,
        simple_global_constraints,
        simple_schedule,
        seed=0,
    )


@pytest.fixture
def student2(
    simple_schedule: list[ScheduleItem],
    simple_global_constraints: list[LinearConstraint],
    course: Course,
    section: Section,
    schedule_item301: ScheduleItem,
    schedule_item611: ScheduleItem,
):
    return RenaissanceMan(
        [[schedule_item301, schedule_item611]],
        [1],
        1,
        1,
        course,
        section,
        simple_global_constraints,
        simple_schedule,
        seed=1,
    )


@pytest.fixture
def student3(
    simple_schedule2: list[ScheduleItem],
    simple_global_constraints: list[LinearConstraint],
    course: Course,
    section: Section,
    schedule_item250_2: ScheduleItem,
    schedule_item301: ScheduleItem,
):
    return RenaissanceMan(
        [[schedule_item250_2, schedule_item301]],
        [1],
        1,
        1,
        course,
        section,
        simple_global_constraints,
        simple_schedule2,
        seed=2,
    )


@pytest.fixture
def bernoullis():
    return np.array([[1, 0, 1], [0, 1, 1]])
