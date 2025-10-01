import argparse, json
from vif.multiagent.run import run_multi_agent
from vif.models.vlm_iface import VLMInterface

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--image', required=True); ap.add_argument('--question', required=True); args = ap.parse_args()
    result = run_multi_agent(image_path=args.image, instruction=args.question, vlm=VLMInterface(mode='stub'))
    print(json.dumps(result, indent=2))
if __name__=='__main__': main()
