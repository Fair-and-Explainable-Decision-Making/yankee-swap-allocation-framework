from typing import List

import numpy as np

from fair.agent import BaseAgent
from fair.constraint import LinearConstraint, PreferenceConstraint
from fair.feature import BaseFeature, Course, Section
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation


class SimulatedAgent(BaseAgent):
    """A randomly generated agent"""

    def __init__(self, constraints: List[LinearConstraint], memoize: bool = True):
        """
        Args:
            constraints (List[LinearConstraint]): constraints to be used in defining valuation
            memoize (bool, optional): Should results be cached. Defaults to True
        """
        super().__init__(ConstraintSatifactionValuation(constraints, memoize))


class RenaissanceMan(SimulatedAgent):
    """Randomly generate a Renaissance man student

    The Renaissance man prefers courses from multiple topics, up to a maximum quantity per topic.
    """

    def __init__(
        self,
        topic_list: List[List[ScheduleItem]],
        max_quantities: List[int],
        lower_max_courses: int,
        upper_max_courses: int,
        course: Course,
        section: Section,
        global_constraints: List[LinearConstraint],
        schedule: List[ScheduleItem],
        seed: int | None = None,
        sparse: bool = False,
        memoize: bool = True,
    ):
        """
        Args:
            topic_list (List[List[ScheduleItem]]): A list of lists of course items, one per topic
            max_quantities (List[int]): The maximum number of courses desired per topic
            lower_max_courses (int): Lower bound for random selection of maximum number of courses (inclusive)
            upper_max_courses (int): Upper bound for random selection of maximum number of courses (inclusive)
            course (Course): Feature for course
            section (Section): Feature for section
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            schedule (List[ScheduleItem], optional): All possible items in the student's schedule. Defaults to None.
            seed (int | None, optional): Random seed. Defaults to None.
            sparse (bool, optional): Should sparse matrices be used for constraints. Defaults to False.
            memoize (bool, optional): Should results be cached. Defaults to True
        """
        rng = np.random.default_rng(seed)

        self.quantities = []
        for max_quant in max_quantities:
            self.quantities.append(rng.integers(0, max_quant + 1))

        self.preferred_topics = []
        self.preferred_courses = []
        for i, quant in enumerate(self.quantities):
            topic = rng.choice(topic_list[i], quant, replace=False).tolist()
            self.preferred_topics.append(topic)
            self.preferred_courses += topic

        self.total_courses = rng.integers(lower_max_courses, upper_max_courses + 1)
        all_courses = [(item.value(course), item.value(section)) for item in schedule]
        self.all_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [all_courses],
            [self.total_courses],
            [course, section],
            sparse,
        )
        undesirable_courses = [
            (item.value(course), item.value(section))
            for item in schedule
            if item not in self.preferred_courses
        ]
        self.undesirable_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [undesirable_courses],
            [0],
            [course, section],
            sparse,
        )
        topic_values = [
            [(item.value(course), item.value(section)) for item in topic]
            for topic in self.preferred_topics
        ]
        self.topic_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            topic_values,
            self.quantities,
            [course, section],
            sparse,
        )

        constraints = global_constraints + [
            self.all_courses_constraint,
            self.undesirable_courses_constraint,
            self.topic_constraint,
        ]

        super().__init__(constraints, memoize)


class SubStudent(SimulatedAgent):
    """Generate a sub-student from a student originally created from a different, larger schedule.
    This function inherits the attributes of the previous student, but redefines preferred_courses and constraints according to the new reduced schedule.
    """

    def __init__(
        self,
        quantities,
        preferred_topics,
        preferred_courses,
        total_courses,
        course: Course,
        section: Section,
        global_constraints: List[LinearConstraint],
        schedule: List[ScheduleItem],
        sparse: bool = False,
        memoize: bool = True,
    ):
        """
        Args:
            quantities (List[int]): The maximum number of courses desired per topic
            preferred_topics (List[List[ScheduleItem]]): A list of lists of desired topics
            preferred_courses (List[ScheduleItem]): A list of preferred courses
            total_courses (int): maximum number of courses (inclusive)
            course (Course): Feature for course
            section (Section): Feature for section
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            schedule (List[ScheduleItem], optional): All possible items in the student's schedule. Defaults to None.
            sparse (bool, optional): Should sparse matrices be used for constraints. Defaults to False.
            memoize (bool, optional): Should results be cached. Defaults to True
        """

        self.quantities = quantities
        self.preferred_topics = preferred_topics
        self.preferred_courses = preferred_courses
        self.total_courses = total_courses

        all_courses = [(item.value(course), item.value(section)) for item in schedule]

        self.all_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [all_courses],
            [self.total_courses],
            [course, section],
            sparse,
        )
        undesirable_courses = [
            (item.value(course), item.value(section))
            for item in schedule
            if item not in self.preferred_courses
        ]
        self.undesirable_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [undesirable_courses],
            [0],
            [course, section],
            sparse,
        )
        topic_values = [
            [(item.value(course), item.value(section)) for item in topic]
            for topic in self.preferred_topics
        ]
        self.topic_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            topic_values,
            self.quantities,
            [course, section],
            sparse,
        )

        constraints = global_constraints + [
            self.all_courses_constraint,
            self.undesirable_courses_constraint,
            self.topic_constraint,
        ]

        super().__init__(constraints, memoize)
