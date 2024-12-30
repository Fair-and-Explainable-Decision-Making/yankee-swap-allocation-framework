import numpy as np

from .agent import BaseAgent
from .allocation import get_bundle_from_allocation_matrix
from .item import ScheduleItem


def precompute_bundles_valuations(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Precompute all agents bundles and all agent valuations for said bundles.
    This is a step necessary to run all envy metrics.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        bundles (list(list[ScheduleItem])): ordered list of agnets bundles
        valuations (type[np.ndarray]): len(agents) x len(agents) matrix, element i,j is agent's i valuation of agent's j bundle under X
    """
    bundles = [
        get_bundle_from_allocation_matrix(X, items, i) for i in range(len(agents))
    ]
    valuations = np.zeros((len(agents), len(agents)))
    for i, agent in enumerate(agents):
        for j, bundle in enumerate(bundles):
            valuations[i, j] = agent.valuation(bundle)
    return bundles, valuations


def EF_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free violations.

    Compare every agent to all other agents, fill EF_matrix where EF_matrix[i,j]=1 if agent of index i
    envies agent of index j, 0 otherwise.

    Returns the number of EF violations, and the number of envious agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF_violations
        int: number of envious agents
    """

    num_agents = len(agents)
    EF_matrix = np.zeros((num_agents, num_agents))

    if valuations is None:
        _, valuations = precompute_bundles_valuations(X, agents, items)

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                EF_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                EF_matrix[j, i] = 1
    return np.sum(EF_matrix > 0), np.sum(np.any(EF_matrix > 0, axis=1))


def EF1_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    bundles: list[list[ScheduleItem]] | None = None,
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free up to one item (EF-1) violations.

    Compare every agent to all other agents, fill EF1_matrix where EF1_matrix[i,j]=1 if agent of index i
    envies agent of index j in the EF1 sense.

    Returns the number of EF-1 violations, and the number of envious agents in the EF1 sense.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        bundles (list(list[ScheduleItem])): List of all agents bundles
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF-1 violations
        int: number of envious agents in the EF-1 sense
    """

    if valuations is None:
        bundles, valuations = precompute_bundles_valuations(X, agents, items)

    num_agents = len(agents)
    EF1_matrix = np.zeros((num_agents, num_agents))

    def there_is_item(i, j):
        for item in range(len(bundles[j])):
            new_bundle = bundles[j].copy()
            new_bundle.pop(item)
            if agents[i].valuation(new_bundle) <= valuations[i][i]:
                return True
        return False

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                if not there_is_item(i, j):
                    EF1_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                if not there_is_item(j, i):
                    EF1_matrix[j, i] = 1
    return np.sum(EF1_matrix > 0), np.sum(np.any(EF1_matrix > 0, axis=1))


def EFX_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    bundles: list[list[ScheduleItem]] | None = None,
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free up to any item (EF-X) violations.

    Compare every agent to all other agents, fill EF1_matrix where EFX_matrix[i,j]=1 if agent of index i
    envies agent of index j in the EF1 sense.

    Returns the number of EF-X violations, and the number of envious agents in the EF-X sense.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem
        bundles (list(list[ScheduleItem])): List of all agents bundles
        valuations (type[np.ndarray]): Valuations of all agents for all bundles under X

    Returns:
        int: number of EF-X violations
        int: number of envious agents in the EF-X sense
    """

    if valuations is None:
        bundles, valuations = precompute_bundles_valuations(X, agents, items)

    num_agents = len(agents)
    EFX_matrix = np.zeros((num_agents, num_agents))

    def for_every_item(i, j):
        for item in range(len(bundles[j])):
            new_bundle = bundles[j].copy()
            new_bundle.pop(item)
            if agents[i].valuation(new_bundle) > valuations[i][i]:
                return False
        return True

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                if not for_every_item(i, j):
                    EFX_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                if not for_every_item(j, i):
                    EFX_matrix[j, i] = 1
    return np.sum(EFX_matrix > 0), np.sum(np.any(EFX_matrix > 0, axis=1))
