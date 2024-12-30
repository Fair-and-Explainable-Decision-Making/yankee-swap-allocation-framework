from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.feature import Course, Section, Slot, Weekday
from fair.item import ScheduleItem, sub_schedule
from fair.simulation import RenaissanceMan, SubStudent


def test_renaissance_man(
    course: Course,
    section: Section,
    slot: Slot,
    weekday: Weekday,
    schedule: list[ScheduleItem],
):
    topic_list = [[schedule[0], schedule[2]], [schedule[4]]]
    quantities = [1, 1]
    lower_max_courses = 1
    upper_max_courses = 2
    global_constraints = [
        CourseTimeConstraint.from_items(schedule, slot, weekday),
        MutualExclusivityConstraint.from_items(schedule, course),
    ]

    # preferred course list does not exceed max quantitity for multiple random configurations
    for i in range(10):
        student = RenaissanceMan(
            topic_list,
            quantities,
            lower_max_courses,
            upper_max_courses,
            course,
            section,
            global_constraints,
            schedule,
            i,
        )
        for j in range(len(quantities)):
            assert len(student.preferred_topics[j]) <= student.quantities[j]

    # student without global constraints can always be fully satisfied
    student = RenaissanceMan(
        topic_list,
        quantities,
        lower_max_courses,
        upper_max_courses,
        course,
        section,
        [],
        schedule,
        0,
    )
    for i, quant in enumerate(student.quantities):
        items = [item for item in schedule if item in student.preferred_topics[i]]
        assert student.value(items) == quant


def test_renaissance_man_memoing(
    course: Course,
    section: Section,
    slot: Slot,
    weekday: Weekday,
    schedule: list[ScheduleItem],
):
    topic_list = [[schedule[0], schedule[2]], [schedule[4]]]
    quantities = [1, 1]
    lower_max_courses = 1
    upper_max_courses = 2
    global_constraints = [
        CourseTimeConstraint.from_items(schedule, slot, weekday),
        MutualExclusivityConstraint.from_items(schedule, course),
    ]

    student_no_memo = RenaissanceMan(
        topic_list,
        quantities,
        lower_max_courses,
        upper_max_courses,
        course,
        section,
        global_constraints,
        schedule,
        memoize=False,
    )
    student_no_memo.value(schedule)

    assert student_no_memo.valuation._value_memo == {}

    student_with_memo = RenaissanceMan(
        topic_list,
        quantities,
        lower_max_courses,
        upper_max_courses,
        course,
        section,
        global_constraints,
        schedule,
    )
    student_with_memo.value(schedule)

    assert len(student_with_memo.valuation._value_memo) > 0

    student_with_memo.valuation.reset()

    assert len(student_with_memo.valuation._value_memo) == 0


def test_sub_student(
    renaissance3: RenaissanceMan,
    schedule: list[ScheduleItem],
    course: Course,
    section: Section,
    slot: Slot,
    weekday: Weekday,
):
    bundle = [item for item in schedule if item.values[0] == "301"]

    reduced_schedule = sub_schedule([bundle])

    course_time_constr = CourseTimeConstraint.from_items(
        reduced_schedule, slot, weekday
    )
    course_sect_constr = MutualExclusivityConstraint.from_items(
        reduced_schedule, course
    )

    new_student = SubStudent(
        renaissance3.quantities,
        [
            [item for item in reduced_schedule if item in topic]
            for topic in renaissance3.preferred_topics
        ],
        [item for item in reduced_schedule if item in renaissance3.preferred_courses],
        renaissance3.total_courses,
        course,
        section,
        [course_time_constr, course_sect_constr],
        reduced_schedule,
    )

    assert len(new_student.preferred_courses) < len(renaissance3.preferred_courses)
    assert new_student.value(reduced_schedule) == renaissance3.value(bundle)
