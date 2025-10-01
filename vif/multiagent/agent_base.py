from abc import ABC, abstractmethod
from typing import Dict, Any
class Agent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name; self.role = role
    @abstractmethod
    def act(self, image_path: str, instruction: str, tools: Dict[str, Any]) -> Dict[str, Any]: ...
