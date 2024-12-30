import os
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from fair.agent import LegacyStudent
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair.stats.survey import Corpus, SingleTopicSurvey

NUM_RAND_SAMP = 100
NUM_SUB_KERNELS = 1
SAMPLE_PER_STUDENT = 10

NUM_STUDENTS = 5
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

surveys = [
    SingleTopicSurvey.from_student(schedule, student.student, 0, 1)
    for student in students
]
corpus = Corpus(surveys)
mbeta = corpus.kde_distribution(SAMPLE_PER_STUDENT, NUM_SUB_KERNELS)

pca = PCA(n_components=2)
data = np.vstack([survey.data() for survey in surveys])
data = np.vstack([data, mbeta.sample(NUM_RAND_SAMP)])
data = pca.fit_transform(np.vstack(data))
data1 = data[:NUM_STUDENTS, :]
data2 = data[NUM_STUDENTS:, :]

plt.scatter(data2[:, 0], data2[:, 1], c="b", alpha=0.25)
plt.scatter(data1[:, 0], data1[:, 1], c="r", alpha=0.25, s=150)
plt.legend(["Simulated", "Student"])
plt.title(
    f"Survey respondents ({NUM_STUDENTS}), sub-kernels ({NUM_SUB_KERNELS}), sub-kernel samples ({SAMPLE_PER_STUDENT})"
)
plt.tick_params(labelbottom=False, labelleft=False)
plt.show()
