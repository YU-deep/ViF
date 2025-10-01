from typing import Dict, Any
from .agent import Agent
from ..models.vlm_iface import VLMInterface

def run_multi_agent(image_path: str, instruction: str, vlm: VLMInterface = None) -> Dict[str, Any]:
    tools = {'vlm': vlm or VLMInterface()}
    agent = Agent(); res = agent.run(image_path=image_path, instruction=instruction, tools=tools)
    return {'difficulty': res.difficulty, 'bucket': res.bucket, 'n_agents': res.n_agents, 'final': res.final, 'all_outputs': res.all_outputs}
