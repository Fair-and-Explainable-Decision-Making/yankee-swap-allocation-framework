import os
from collections import defaultdict

import pandas as pd

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, round_robin, serial_dictatorship
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.envy import (
    EF_1_agents,
    EF_1_count,
    EF_agents,
    EF_count,
    EF_X_agents,
    EF_X_count,
)
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.metrics import PMMS_violations, leximin, nash_welfare, utilitarian_welfare
from fair.optimization import StudentAllocationProgram
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 3
MAX_COURSES_PER_TOPIC = 5
LOWER_MAX_COURSES_TOTAL = 1
UPPER_MAX_COURSES_TOTAL = 5
EXCEL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
)
SPARSE = False
FIND_OPTIMAL = True

# load schedule as DataFrame
with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
    df = pd.read_excel(fd)

# construct features from DataFrame
course = Course(df["Catalog"].astype(str).unique().tolist())

time_ranges = df["Mtg Time"].dropna().unique()
slot = Slot.from_time_ranges(time_ranges, "15T")
weekday = Weekday()

section = Section(df["Section"].dropna().unique().tolist())
features = [course, slot, weekday, section]

# construct schedule
schedule = []
topic_map = defaultdict(list)
for idx, (_, row) in enumerate(df.iterrows()):
    crs = str(row["Catalog"])
    slt = slots_for_time_range(row["Mtg Time"], slot.times)
    sec = row["Section"]
    capacity = row["CICScapacity"]
    dys = tuple([day.strip() for day in row["zc.days"].split(" ")])
    item = ScheduleItem(features, [crs, slt, dys, sec], index=idx, capacity=capacity)
    schedule.append(item)
    topic_map[row["Categories"]].append(item)

topics = [topic for topic in topic_map.values()]

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, weekday, SPARSE)
course_sect_constr = MutualExclusivityConstraint.from_items(schedule, course, SPARSE)

# randomly generate students
students = []
for i in range(NUM_STUDENTS):
    student = RenaissanceMan(
        topics,
        [min(len(topic), MAX_COURSES_PER_TOPIC) for topic in topics],
        LOWER_MAX_COURSES_TOTAL,
        UPPER_MAX_COURSES_TOTAL,
        course,
        section,
        [course_time_constr, course_sect_constr],
        schedule,
        seed=i,
        sparse=SPARSE,
    )
    legacy_student = LegacyStudent(student, student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    students.append(legacy_student)

X_YS, _, _ = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS nash welfare: ", nash_welfare(X_YS, students, schedule))
print("YS leximin vector: ", leximin((X_YS), students, schedule))
print("YS EF_count: ", EF_count(X_YS, students, schedule))
print("YS EF_agents: ", EF_agents(X_YS, students, schedule))
print("YS EF_1_count: ", EF_1_count(X_YS, students, schedule))
print("YS EF_1_agents: ", EF_1_agents(X_YS, students, schedule))
print("YS EF_X_count: ", EF_X_count(X_YS, students, schedule))
print("YS EF_X_agents: ", EF_X_agents(X_YS, students, schedule))
print("YS PMMS violations (total, agents): ", PMMS_violations(X_YS, students, schedule))

X_SD = serial_dictatorship(students, schedule)
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))
print("SD nash welfare: ", nash_welfare(X_SD, students, schedule))
print("SD leximin vector: ", leximin(X_SD, students, schedule))
print("SD EF_count: ", EF_count(X_SD, students, schedule))
print("SD EF_agents: ", EF_agents(X_SD, students, schedule))
print("SD EF_1_count: ", EF_1_count(X_SD, students, schedule))
print("SD EF_1_agents: ", EF_1_agents(X_SD, students, schedule))
print("SD EF_X_count: ", EF_X_count(X_SD, students, schedule))
print("SD EF_X_agents: ", EF_X_agents(X_SD, students, schedule))
print("SD PMMS violations (total, agents): ", PMMS_violations(X_SD, students, schedule))

X_RR = round_robin(students, schedule)
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR nash welfare: ", nash_welfare(X_RR, students, schedule))
print("RR leximin vector: ", leximin(X_RR, students, schedule))
print("RR EF_count: ", EF_count(X_RR, students, schedule))
print("RR EF_agents: ", EF_agents(X_RR, students, schedule))
print("RR EF_1_count: ", EF_1_count(X_RR, students, schedule))
print("RR EF_1_agents: ", EF_1_agents(X_RR, students, schedule))
print("RR EF_X_count: ", EF_X_count(X_RR, students, schedule))
print("RR EF_X_agents: ", EF_X_agents(X_RR, students, schedule))
print("RR PMMS violations (total, agents): ", PMMS_violations(X_RR, students, schedule))


if FIND_OPTIMAL:
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()
    X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()
    print("ILP utilitarian welfare: ", utilitarian_welfare(X_ILP, students, schedule))
    print("ILP nash welfare: ", nash_welfare(X_ILP, students, schedule))
    print("ILP leximin vector: ", leximin(X_ILP, students, schedule))
    print("ILP EF_count: ", EF_count(X_ILP, students, schedule))
    print("ILP EF_agents: ", EF_agents(X_ILP, students, schedule))
    print("ILP EF_1_count: ", EF_1_count(X_ILP, students, schedule))
    print("ILP EF_1_agents: ", EF_1_agents(X_ILP, students, schedule))
    print("ILP EF_X_count: ", EF_X_count(X_ILP, students, schedule))
    print("ILP EF_X_agents: ", EF_X_agents(X_ILP, students, schedule))
    print(
        "ILP PMMS violations (total, agents): ",
        PMMS_violations(X_ILP, students, schedule),
    )
