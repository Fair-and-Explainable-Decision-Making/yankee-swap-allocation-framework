import numpy as np

from ..item import ScheduleItem
from ..simulation import RenaissanceMan
from . import Correlation, Mean, Shape, bernoulli_samples, mBetaApprox, mBetaMixture


class BaseSurvey:
    """Abstract survey class"""

    pass


class SingleTopicSurvey(BaseSurvey):
    """A single-topic survey"""

    @staticmethod
    def from_student(
        schedule: list[ScheduleItem],
        student: RenaissanceMan,
        response_lower_extent: int,
        response_upper_extent: int,
    ) -> "SingleTopicSurvey":
        """Create a survey from an existing RenaissanceMan student

        Ignores divisions by topic, pooling all courses into one list.

        Args:
            schedule (list[ScheduleItem]): Schedule from which student was created
            student (RenaissanceMan): Student from which survey will be created
            response_lower_extent (int): Minimum possible response value
            response_upper_extent (int): Maximum possible response value

        Returns:
            SingleTopicSurvey: A new survey object
        """
        responses = [student.valuation.independent([item]) for item in schedule]

        return SingleTopicSurvey(
            schedule,
            responses,
            student.total_courses,
            response_lower_extent,
            response_upper_extent,
        )

    def __init__(
        self,
        schedule: list[ScheduleItem],
        responses: list[int],
        limit: int,
        response_lower_extent: int,
        response_upper_extent: int,
    ) -> None:
        """A survey that groups all responses into a single topic

        Args:
            schedule (list[ScheduleItem]): Schedule from which to draw course information
            responses (list[int]): Student survey responses
            limit (int): Total courses desired
            response_lower_extent (int): Minimum possible response value
            response_upper_extent (int): Maximum possible response value
        """
        self.schedule = schedule
        self.response_map = {schedule[i]: responses[i] for i in range(len(schedule))}
        self.limit = limit
        self.response_lower_extent = response_lower_extent
        self.response_upper_extent = response_upper_extent
        self.m = len(self.schedule)

    def data(self) -> np.ndarray:
        """Create data vector from responses

        Raises:
            ValueError: Upper extent must be greater than lower extent for normalization

        Returns:
            np.ndarray: Vector of normalized responses
        """
        data = np.array([self.response_map[item] for item in self.schedule])
        sm = data.sum()

        if self.response_upper_extent <= self.response_lower_extent:
            raise ValueError(
                "Upper extent must be greater than lower extent for normalization"
            )

        # normalize data
        data = (data - self.response_lower_extent) / (
            self.response_upper_extent - self.response_lower_extent
        )

        return data.reshape((1, self.m))


class Corpus:
    """A collection of surveys"""

    def __init__(
        self, surveys: list[BaseSurvey], rng: np.random.Generator | None = None
    ):
        """
        Args:
            surveys (list[BaseSurvey]): Survey list
            rng (np.random.Generator | None): Random number generator. Defaults to None.
        """
        self.rng = np.random.default_rng(None) if rng is None else rng
        self.surveys = surveys

    def _valid(self):
        """Schedule items in surveys must match

        Returns:
            bool: True if schedule items match, False otherwise
        """
        if len(self.surveys) < 1:
            return False

        base = self.surveys[0]
        for survey in self.surveys[1:]:
            if len(base.schedule) != len(survey.schedule):
                return False
            for i in range(len(base.schedule)):
                if base.schedule[i] != survey.schedule[i]:
                    return False

        return True

    def distribution(self) -> mBetaApprox:
        """Create an mBeta distribution from the survey data

        Raises:
            ValueError: Corpus must pass validation

        Returns:
            mBetaApprox: Approximate mBeta distribution
        """
        if not self._valid():
            raise ValueError("Invalid Corpus for generating distribution")

        m = self.surveys[0].m
        R = Correlation(m)
        nu = Shape(0.001)
        mu = Mean(m)
        mbeta = mBetaApprox(R, mu, nu, self.rng)
        for survey in self.surveys:
            sample = bernoulli_samples(survey.data(), self.rng)
            mbeta.update(sample)

        return mbeta

    def kde_distribution(self, n: int = 1, k: int = 1) -> mBetaMixture:
        """Create a mixture of mBeta distributions, one for each survey

        Corresponding to each survey, k sub-kernels are generated, each drawing
        n Bernoulli samples according to survey data. This defines an mBetaMixture
        object per survey. Those mBetaMixture objecs are combined as the kernels in
        a second-level mBetaMixture object.

        Args:
            n (int, optional): Number of samples per sub-kernel. Defaults to 1.
            k (int, optional): Sub-kernals per survey. Defaults to 1.

        Raises:
            ValueError: Corpus must pass validation

        Returns:
            mBetaMixture: Approximate mBeta mixture distribution
        """
        if not self._valid():
            raise ValueError("Invalid Corpus for generating distribution")

        mbeta_kdes = []
        for survey in self.surveys:
            mbetas = []
            for i in range(k):
                m = self.surveys[0].m
                R = Correlation(m)
                nu = Shape(0.001)
                mu = Mean(m)
                mbeta = mBetaApprox(R, mu, nu, self.rng)
                sample = bernoulli_samples(survey.data(), self.rng, n)
                mbeta.update(sample)
                mbetas.append(mbeta)
            mbeta_kdes.append(mBetaMixture(mbetas, self.rng))

        return mBetaMixture(mbeta_kdes, self.rng)
