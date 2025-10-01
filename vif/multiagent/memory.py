from typing import List, Dict, Any
class EpisodicMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity; self._mem: List[Dict[str, Any]] = []
    def add(self, item: Dict[str, Any]):
        self._mem.append(item)
        if len(self._mem) > self.capacity:
            self._mem.pop(0)
    def last(self, k: int = 1) -> List[Dict[str, Any]]:
        return self._mem[-k:]
