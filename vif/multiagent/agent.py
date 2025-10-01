from dataclasses import dataclass
from typing import Dict, Any
from .difficulty import estimate_difficulty
from .allocation import map_difficulty_to_allocation
from .router import build_agents
from .consensus import majority_vote
from .memory import EpisodicMemory

@dataclass
class MetaResult:
    difficulty: float
    bucket: str
    n_agents: int
    final: Dict[str, Any]
    all_outputs: Any

class Agent:
    def __init__(self): self.memory = EpisodicMemory(capacity=200)
    def run(self, image_path: str, instruction: str, tools: Dict[str, Any]) -> MetaResult:
        diff = estimate_difficulty(instruction)
        alloc = map_difficulty_to_allocation(diff.buckets)
        agents = build_agents(alloc.roles)
        outputs = [a.act(image_path=image_path, instruction=instruction, tools=tools) for a in agents]
        final = majority_vote(outputs)
        self.memory.add({'image': image_path, 'instruction': instruction, 'final': final, 'all': outputs})
        return MetaResult(difficulty=diff.score, bucket=diff.buckets, n_agents=alloc.n_agents, final=final, all_outputs=outputs)
