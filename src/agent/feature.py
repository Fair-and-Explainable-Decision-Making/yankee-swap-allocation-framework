class DomainError(Exception):
    pass


class FeatureError(Exception):
    pass


class BaseFeature:
    def __init__(self, name: str, domain: list):
        self.name = name
        self.domain = domain

    def __repr__(self):
        return f"{self.name}: [{self.domain[0]} ... {self.domain[-1]}]"


class Course(BaseFeature):
    def __init__(self, domain):
        super().__init__("course", domain)