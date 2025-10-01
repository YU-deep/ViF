from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Message:
    role: str
    content: str
    meta: Optional[Dict[str, Any]] = None
