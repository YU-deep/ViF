"""Example evaluation script (stub) for running multi-agent inference with ViF.
Replace the dummy parts with real VLM inference code and dataset/evaluator.
"""
import torch
from .src.vif import ViFModule

def run_single_agent(vlm_model, image, instruction, vif_module=None):
    attn_maps = torch.rand(32, 196)
    output_text = "<answer>"
    return output_text, attn_maps

def run_multi_agent_pipeline(image, instruction, agents=5, vif_module=None):
    history = []
    prev_text = None
    for t in range(agents):
        instr = instruction if prev_text is None else (instruction + ' Previous: ' + prev_text)
        out, attn = run_single_agent(None, image, instr, vif_module=vif_module)
        if vif_module is not None:
            sel = vif_module.select_relay_tokens(attn)
        prev_text = out
        history.append(out)
    return history



def main():
    vif = ViFModule(feat_dim=768, num_relay=32)
    image = None
    instr = "Describe objects on the plate."
    hist = run_multi_agent_pipeline(image, instr, agents=8, vif_module=vif)
    print('Multi-agent outputs:', hist)

if __name__ == '__main__':
    main()
