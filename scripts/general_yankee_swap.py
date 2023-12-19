import os
from collections import defaultdict

import pandas as pd

from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, slots_for_time_range
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan

NUM_STUDENTS = 100
MAX_COURSES_PER_TOPIC = 5
MAX_COURSES_TOTAL = 5
EXCEL_SCHEDULE_PATH = os.path.join(
    os.path.dirname(__file__), "../resources/fall2023schedule-2-cat.xlsx"
)

# load schedule as DataFrame
with open(EXCEL_SCHEDULE_PATH, "rb") as fd:
    df = pd.read_excel(fd)

# construct features from DataFrame
course = Course(df["Catalog"].astype(str).unique().tolist())

time_ranges = df["Mtg Time"].dropna().unique()
slot = Slot.from_time_ranges(time_ranges, "15T")

section = Section(df["Section"].dropna().unique().tolist())
features = [course, slot, section]

# construct schedule
schedule = []
topic_map = defaultdict(set)
for _, row in df.iterrows():
    crs = str(row["Catalog"])
    topic_map[row["Categories"]].add(crs)
    slt = slots_for_time_range(row["Mtg Time"], slot.times)
    sec = row["Section"]
    schedule.append(ScheduleItem(features, [crs, slt, sec]))

topics = [list(courses) for courses in topic_map.values()]

# global constraints
course_time_constr = CourseTimeConstraint.from_items(schedule, slot, [course, slot])
course_sect_constr = MutualExclusivityConstraint.from_items(
    schedule, course, [course, section]
)

# randomly generate students
students = []
for i in range(NUM_STUDENTS):
    student = RenaissanceMan(
        topics,
        [min(len(topic), MAX_COURSES_PER_TOPIC) for topic in topics],
        MAX_COURSES_TOTAL,
        course,
        [course_time_constr, course_sect_constr],
        seed=i,
    )
    students.append(LegacyStudent(student, student.all_courses_constraint))

general_yankee_swap(students, schedule)