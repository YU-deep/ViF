from typing import Dict, Any
from .agent_base import Agent
ROLE_PROMPTS = {
 'Perception': 'You are a precise visual describer. Focus on salient objects and attributes.',
 'Localization': 'You reason about spatial relations (left/right/top, distances, relative positions).',
 'Counting': 'You are a careful counter. Count distinct instances and justify briefly.',
 'OCR': 'You read visible text in the image. Return text spans verbatim if possible.',
 'Commonsense': 'You connect visual cues with commonsense/world knowledge to answer why/how.'}
class VisionWorker(Agent):
    def __init__(self, name: str, role: str): super().__init__(name, role)
    def act(self, image_path: str, instruction: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        vlm = tools['vlm']
        sys = ROLE_PROMPTS.get(self.role, 'You are a helpful assistant for visual tasks.')
        prompt = f"{sys}\nImage: {image_path}\nTask: {instruction}\nPlease answer and provide a short rationale. Also output a confidence between 0 and 1."
        out = vlm.generate(image_path=image_path, prompt=prompt)
        return {'agent': self.name, 'role': self.role, 'answer': out.get('text',''), 'rationale': out.get('rationale',''), 'confidence': float(out.get('confidence',0.5))}
