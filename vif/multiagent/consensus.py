from typing import List, Dict
from collections import defaultdict

def majority_vote(outputs: List[Dict]) -> Dict:
    buckets = defaultdict(list)
    for o in outputs:
        key = (o.get('answer','') or '').strip().lower(); buckets[key].append(o)
    best, best_score = None, -1.0
    for k, lst in buckets.items():
        score = sum(x.get('confidence',0.0) for x in lst) + 0.2*len(lst)
        if score > best_score:
            best_score = score; best = dict(lst[0]); best['consensus_support']=len(lst); best['consensus_score']=score
    return best or (outputs[0] if outputs else {'answer':'', 'confidence':0.0})
