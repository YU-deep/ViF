"""Integration"""
import torch
from transformers import AutoProcessor

"""
load model here, below is an example of LLaVA-NeXT
"""
try:
    from transformers import LlavaNextForConditionalGeneration as Model
except Exception:
    try:
        from transformers import LlavaForConditionalGeneration as Model
    except Exception:
        Model = None

import warnings



def load(model_name, device):
    processor = AutoProcessor.from_pretrained(model_name)
    if Model is None:
        raise RuntimeError("Could not import model class from transformers.")
    kwargs = {}
    if torch.cuda.is_available():
        kwargs['torch_dtype'] = torch.float16
        try:
            model = Model.from_pretrained(model_name, device_map='auto', **kwargs)
        except Exception as e:
            warnings.warn(f"device_map auto failed: {e}; loading to device manually")
            model = Model.from_pretrained(model_name, **kwargs)
            model.to(device)
    else:
        model = Model.from_pretrained(model_name)
    try:
        model.config.output_attentions = True
    except Exception:
        warnings.warn('Could not set model.config.output_attentions; pass output_attentions=True at forward time.')
    return model, processor


def prepare_inputs(processor, image_list, question_text, device='cuda'):
    prompt = f"USER: {question_text}\nASSISTANT:"
    inputs = processor(images=image_list, text=prompt, return_tensors='pt')
    for k,v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs



def extract_vision_token_features(model, inputs):
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(**inputs, output_attentions=True, return_dict=True)
        except TypeError:
            outputs = model(**inputs, output_attentions=True)
    result = {}
    cross_attns = None
    if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
        cross_attns = outputs.cross_attentions
    else:
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            cross_attns = outputs.attentions
    result['cross_attentions'] = cross_attns
    vision_feats = None
    for key in ['vision_last_hidden_state', 'encoder_last_hidden_state', 'image_embeds', 'last_hidden_state']:
        if hasattr(outputs, key):
            vision_feats = getattr(outputs, key)
            break
    if vision_feats is None:
        for attr in ['vision_tower', 'vision_encoder', 'encoder']:
            if hasattr(model, attr):
                try:
                    vision_module = getattr(model, attr)
                    if 'pixel_values' in inputs:
                        vout = vision_module(pixel_values=inputs['pixel_values'], output_attentions=False, return_dict=True)
                        if hasattr(vout, 'last_hidden_state'):
                            vision_feats = vout.last_hidden_state
                            break
                except Exception:
                    continue
    result['vision_feats'] = vision_feats
    return result


def get_decoder_cross_attention_during_generation(model, inputs, max_new_tokens=64, device='cuda'):
    model.eval()
    with torch.no_grad():
        try:
            gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     output_attentions=True, return_dict_in_generate=True)
        except TypeError:
            gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    cross_attns = None
    if hasattr(gen_out, 'cross_attentions'):
        cross_attns = gen_out.cross_attentions
    return gen_out, cross_attns
