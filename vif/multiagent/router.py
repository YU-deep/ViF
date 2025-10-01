from typing import List
from .worker_agent import VisionWorker

def build_agents(role_list: List[str]) -> List[VisionWorker]:
    return [VisionWorker(name=f'agent_{i+1}', role=r) for i, r in enumerate(role_list)]
