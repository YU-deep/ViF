import random
class VLMInterface:
    def __init__(self, mode: str = 'stub'): self.mode = mode
    def generate(self, image_path: str, prompt: str):
        if self.mode == 'stub':
            seed = hash((image_path, prompt)) % (2**32); rng = random.Random(seed)
            conf = 0.5 + 0.5*rng.random(); txt = 'object detected' if 'describe' in prompt.lower() else 'analysis complete'
            return {'text': txt, 'rationale': 'stub rationale', 'confidence': conf}
        else:
            raise NotImplementedError('Implement your real VLM backend here.')
