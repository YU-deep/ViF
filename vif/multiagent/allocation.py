from dataclasses import dataclass

@dataclass
class Allocation:
    n_agents: int
    roles: list

DEFAULT_ROLE_POOL = ['Perception','Localization','Counting','OCR','Commonsense']

def map_difficulty_to_allocation(bucket: str) -> Allocation:
    n = 1 if bucket=='easy' else (3 if bucket=='medium' else 5)
    return Allocation(n_agents=n, roles=DEFAULT_ROLE_POOL[:n])
