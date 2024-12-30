from fair.agent import LegacyStudent
from fair.allocation import (
    general_yankee_swap,
    general_yankee_swap_E,
    round_robin,
    serial_dictatorship,
)
from fair.feature import Course
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan


def test_general_yankee_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.preferred_courses, course)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.preferred_courses, course)

    X, _, _ = general_yankee_swap([leg_student1, leg_student2], schedule)
    alloc1 = X[:, 0]
    alloc2 = X[:, 1]

    # quantity allocated courses does not exceed limit
    assert sum(alloc1) <= renaissance1.total_courses
    assert sum(alloc2) <= renaissance2.total_courses

    # only preferred courses are allocated
    courses1 = [schedule[i] for i in range(len(alloc1)) if alloc1[i] == 1]
    courses2 = [schedule[i] for i in range(len(alloc2)) if alloc2[i] == 1]
    assert set(courses1) <= set(renaissance1.preferred_courses)
    assert set(courses2) <= set(renaissance2.preferred_courses)


def test_general_yankee_swap_E(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.preferred_courses, course)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.preferred_courses, course)

    X, _, _ = general_yankee_swap_E([leg_student1, leg_student2], schedule)
    alloc1 = X[:, 0]
    alloc2 = X[:, 1]

    # quantity allocated courses does not exceed limit
    assert sum(alloc1) <= renaissance1.total_courses
    assert sum(alloc2) <= renaissance2.total_courses

    # only preferred courses are allocated
    courses1 = [schedule[i] for i in range(len(alloc1)) if alloc1[i] == 1]
    courses2 = [schedule[i] for i in range(len(alloc2)) if alloc2[i] == 1]
    assert set(courses1) <= set(renaissance1.preferred_courses)
    assert set(courses2) <= set(renaissance2.preferred_courses)


def test_round_robin_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.preferred_courses, course)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.preferred_courses, course)

    X = round_robin([leg_student1, leg_student2], schedule)
    alloc1 = X[:, 0]
    alloc2 = X[:, 1]

    # quantity allocated courses does not exceed limit
    assert sum(alloc1) <= renaissance1.total_courses
    assert sum(alloc2) <= renaissance2.total_courses

    # only preferred courses are allocated
    courses1 = [schedule[i] for i in range(len(alloc1)) if alloc1[i] == 1]
    courses2 = [schedule[i] for i in range(len(alloc2)) if alloc2[i] == 1]
    assert set(courses1) <= set(renaissance1.preferred_courses)
    assert set(courses2) <= set(renaissance2.preferred_courses)


def test_serial_dictatorship_swap(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
):
    leg_student1 = LegacyStudent(renaissance1, renaissance1.preferred_courses, course)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.preferred_courses, course)

    X = serial_dictatorship([leg_student1, leg_student2], schedule)
    alloc1 = X[:, 0]
    alloc2 = X[:, 1]

    # quantity allocated courses does not exceed limit
    assert sum(alloc1) <= renaissance1.total_courses
    assert sum(alloc2) <= renaissance2.total_courses

    # only preferred courses are allocated
    courses1 = [schedule[i] for i in range(len(alloc1)) if alloc1[i] == 1]
    courses2 = [schedule[i] for i in range(len(alloc2)) if alloc2[i] == 1]
    assert set(courses1) <= set(renaissance1.preferred_courses)
    assert set(courses2) <= set(renaissance2.preferred_courses)
