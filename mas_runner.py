"""Simple orchestrator for MAS topologies (linear, circular, random, layered)."""
import random
def build_topology(structure, num_agents):
    if structure == 'linear':
        return [(i, i+1) for i in range(num_agents-1)]
    if structure == 'circular':
        return [(i, (i+1)%num_agents) for i in range(num_agents)]
    if structure == 'random':
        edges = []
        for i in range(num_agents):
            to = random.randrange(num_agents)
            edges.append((i,to))
        return edges
    if structure == 'layered':
        edges = []
        mid = num_agents//2
        for i in range(mid-1):
            edges.append((i,i+1))
        for i in range(mid, num_agents-1):
            edges.append((i,i+1))
        return edges
    raise ValueError('unknown structure')


if __name__ == '__main__':
    print('Example topologies:')
    for s in ['linear','circular','random','layered']:
        print(s, build_topology(s, 6))
