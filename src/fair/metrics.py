import copy

import numpy as np

from .agent import BaseAgent, LegacyStudent
from .allocation import general_yankee_swap_E, get_bundle_from_allocation_matrix
from .constraint import CourseTimeConstraint, MutualExclusivityConstraint
from .item import ScheduleItem, sub_schedule
from .simulation import SubStudent


def utilitarian_welfare(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute utilitarian social welfare (USW)

    Calculates the average of utilities across all agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        float: USW / len(agents)
    """
    util = 0
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        util += val
    return util / (len(agents))


def nash_welfare(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute Nash social welfare (NSW)

    Calculates the number of agents with 0 utility and product of utilities across all
    agents with utility > 0.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        int: number of agents with utility 0
        float: n-root of NSW
    """
    util = 0
    num_zeros = 0
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        if val == 0:
            num_zeros += 1
        else:
            util += np.log(val)
    return num_zeros, np.exp(util / (len(agents) - num_zeros))


def leximin(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute Leximin vector, i.e. vector with agents utilities, sorted in decreasing order

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    valuations = []
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        valuations.append(val)
    valuations.sort()
    valuations.reverse()
    return valuations


# FUNCTIONS TO COMPUTE PAIRWISE MAXIMIN SHARE


def yankee_swap_sub_problem(
    agent: type[BaseAgent],
    new_schedule: list[ScheduleItem],
    course_strings: list[str],
):
    """Given an agent and information of a reduced schedule (new_schedule, course_strings, course), compute their MMS for the reduced problem,
    considering 2 identical agents competing for the items in the reduced schedule.
    We do this by computing a leximin allocation through yankee swap.

    Args:
        agent (type[BaseAgent]): Agent from the class BaseAgent
        new_schedule (list[ScheduleItem]): Items from class BaseItem, new reduced schedule
        course_strings (list[str]): List of course strings of the new schedule
        course (type[Course]): Course instance of the new schedule

    Returns:
        int: Agent's MMS for the subproblem
    """
    course, slot, weekday, section = new_schedule[0].features

    course_time_constr = CourseTimeConstraint.from_items(new_schedule, slot, weekday)
    course_sect_constr = MutualExclusivityConstraint.from_items(new_schedule, course)
    preferred = agent.preferred_courses
    new_student = SubStudent(
        agent.student.quantities,
        [
            [item for item in topic if item in new_schedule]
            for topic in agent.student.preferred_topics
        ],
        [item for item in preferred if item in new_schedule],
        agent.student.total_courses,
        course,
        section,
        [course_time_constr, course_sect_constr],
        new_schedule,
    )

    legacy_student = LegacyStudent(new_student, new_student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    sub_student = legacy_student

    X_sub, _, _ = general_yankee_swap_E([sub_student, sub_student], new_schedule)

    bundle_1 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 0)
    bundle_2 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 1)

    return min([sub_student.valuation(bundle_1), sub_student.valuation(bundle_2)])


def pairwise_maximin_share(
    agent1: type[BaseAgent],
    agent2: type[BaseAgent],
    current_bundle_1: list[ScheduleItem],
    current_bundle_2: list[ScheduleItem],
):
    """Given two agents and their current bundles, compute their Pairwise Maximin Share (PMMS)

    Args:
        agent1 (type[BaseAgent]): First agent
        agent2 (type[BaseAgent]): Second agent
        current_bundle_1 (list[ScheduleItem]): first agent's current bundle
        current_bundle_2 (list[ScheduleItem]): second agent's current bundle

    Returns:
        PMMS[BaseAgent] (type[int]): for agents 1 and 2, return their PMMS for the subproblem
    """

    PMMS = {}

    bundle_1 = copy.deepcopy([sched for sched in current_bundle_1])
    for sched in bundle_1:
        sched.capacity = 1
    bundle_2 = copy.deepcopy([sched for sched in current_bundle_2])
    for sched in bundle_2:
        sched.capacity = 1

    new_schedule = sub_schedule([bundle_1, bundle_2])
    course_strings = sorted([item.values[0] for item in new_schedule])

    PMMS[agent1] = yankee_swap_sub_problem(agent1, new_schedule, course_strings)
    PMMS[agent2] = yankee_swap_sub_problem(agent2, new_schedule, course_strings)

    return PMMS


def PMMS_violations(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute number of violations of the Pairwise Maximin Share (PMMS) for an allocation X

    Compare every agent to all agents of higher index and determine whether they receive their PMMS.
    Runs intermediate functions to compute PMMS and returns tuple with the number of comparison which did not comply with the PMMS,
    and the number of agents that for at least one comparison, did not receive their PMMS.

     Args:
         X (type[np.ndarray]): Allocation matrix
         agents (list[BaseAgent]): Agents from class BaseAgent
         schedule (list[ScheduleItem]): Items from class BaseItem

     Returns:
         int: Number of PMMS violations
         int: Number of agents who did not receive their PMMS in every comparison
    """
    PMMS_matrix = np.zeros((len(agents), len(agents)))
    for i, student_1 in enumerate(agents):
        bundle_1 = get_bundle_from_allocation_matrix(X, items, i)

        for j in range(i + 1, len(agents)):
            student_2 = agents[j]
            bundle_2 = get_bundle_from_allocation_matrix(X, items, j)

            if len(bundle_1) == 0 and len(bundle_2) == 0:
                continue

            PMMS = pairwise_maximin_share(student_1, student_2, bundle_1, bundle_2)
            PMMS_matrix[i, j] = student_1.valuation(bundle_1) - PMMS[student_1]
            PMMS_matrix[j, i] = student_2.valuation(bundle_2) - PMMS[student_2]

    return np.sum(PMMS_matrix < 0), np.sum(np.any(PMMS_matrix < 0, axis=1))
