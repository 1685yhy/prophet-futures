import networkx as nx
from typing import Dict, Any


def build_futures_causal_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    
    G.add_nodes_from([
        '货币政策', '财政政策', '宏观经济', '行业政策',
        '供应', '需求', '库存', '进出口',
        '原油', '铁矿石', '螺纹钢', '焦炭',
        '汇率', '利率', '通胀', '流动性',
        '期货价格', '现货价格', '基差', '持仓结构'
    ])
    
    edges = [
        ('货币政策', '利率'),
        ('货币政策', '流动性'),
        ('财政政策', '宏观经济'),
        ('宏观经济', '需求'),
        ('行业政策', '供应'),
        ('供应', '库存'),
        ('需求', '库存'),
        ('库存', '基差'),
        ('基差', '期货价格'),
        ('现货价格', '期货价格'),
        ('原油', '化工品'),
        ('铁矿石', '螺纹钢'),
        ('螺纹钢', '焦炭'),
        ('汇率', '进出口'),
        ('进出口', '供应'),
        ('通胀', '货币政策'),
        ('利率', '期货价格'),
        ('流动性', '期货价格'),
        ('持仓结构', '期货价格')
    ]
    
    G.add_edges_from(edges)
    
    return G


def query_causal_graph(event_type: str, target: str) -> Dict[str, Any]:
    G = build_futures_causal_graph()
    
    if event_type not in G.nodes or target not in G.nodes:
        return {
            'causal_chain': [],
            'confidence': 0.3,
            'effect_strength': 'WEAK',
            'explanation': '未找到明确的因果路径'
        }
    
    try:
        paths = list(nx.all_simple_paths(G, event_type, target))
        if paths:
            shortest_path = min(paths, key=len)
            chain_str = ' → '.join(shortest_path)
            
            effect_mapping = {
                1: ('STRONG', 0.85),
                2: ('MODERATE', 0.7),
                3: ('MODERATE', 0.55),
                4: ('WEAK', 0.4),
                5: ('WEAK', 0.3)
            }
            length = len(shortest_path) - 1
            strength, confidence = effect_mapping.get(length, ('WEAK', 0.25))
            
            return {
                'causal_chain': shortest_path,
                'chain_description': chain_str,
                'effect_strength': strength,
                'confidence': confidence,
                'explanation': f'从{event_type}到{target}的因果路径：{chain_str}'
            }
    except Exception:
        pass
    
    return {
        'causal_chain': [],
        'confidence': 0.2,
        'effect_strength': 'WEAK',
        'explanation': '无法确定因果关系'
    }


def do_intervention(graph: nx.DiGraph, intervention: str) -> float:
    affected_nodes = list(graph.successors(intervention))
    effect = 0.0
    
    for node in affected_nodes:
        effect += 0.1 * (1 if '价格' in node else 0.5)
    
    return effect