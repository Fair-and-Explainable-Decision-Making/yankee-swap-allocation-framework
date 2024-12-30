import numpy as np

from fair.agent import BaseAgent, LegacyStudent
from fair.allocation import general_yankee_swap
from fair.constraint import PreferenceConstraint
from fair.feature import Course
from fair.item import ScheduleItem
from fair.metrics import utilitarian_welfare
from fair.optimization import IntegerLinearProgram, StudentAllocationProgram
from fair.simulation import RenaissanceMan
from fair.valuation import ConstraintSatifactionValuation


def test_integer_linear(
    renaissance1: RenaissanceMan,
    renaissance2: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
):
    agents = [renaissance1, renaissance2]
    leg_student1 = LegacyStudent(renaissance1, renaissance1.preferred_courses, course)
    leg_student2 = LegacyStudent(renaissance2, renaissance2.preferred_courses, course)
    leg_agents = [leg_student1, leg_student2]

    X, _, _ = general_yankee_swap([leg_student1, leg_student2], schedule)
    util = utilitarian_welfare(X, leg_agents, schedule)

    program = IntegerLinearProgram(agents).compile()
    ind = program.convert_allocation(X)

    # ensure that allocation returned by YS violates no constraints
    assert not np.sum(program.A @ ind > program.b) > 0

    # since allocation X should be non-redundant, USW should equal sum(ind) / len(agents)
    assert np.sum(ind) / len(agents) == util

    opt_alloc = program.formulateUSW().solve()

    # allocation should be feasible
    assert not np.sum(program.A @ opt_alloc > program.b.T) > 0

    # YS solution should not exceed ilp solution
    assert sum(opt_alloc) / len(agents) >= util


def test_capacity_constraint(course: Course):
    # course capacities are 1 by default
    schedule = [
        ScheduleItem([course], ["250"], 0),
        ScheduleItem([course], ["301"], 1),
        ScheduleItem([course], ["611"], 2),
    ]
    constraint = PreferenceConstraint.from_item_lists(
        schedule, [[("250",), ("301",), ("611",)]], [3], [course]
    )
    valuation = ConstraintSatifactionValuation([constraint])

    # two students with exactly the same preferences
    agents = [BaseAgent(valuation), BaseAgent(valuation)]

    program = StudentAllocationProgram(agents, schedule).compile()
    opt_alloc = program.formulateUSW().solve()

    # only three courses total could have been allocated
    assert np.sum(opt_alloc) == 3

    # change course capacities to 2
    for item in schedule:
        item.capacity = 2

    program = StudentAllocationProgram(agents, schedule).compile()
    opt_alloc = program.formulateUSW().solve()

    # now it's possible to allocate each of the three courses twice
    assert np.sum(opt_alloc) == 6
