import networkx as nx
from typing import Dict, Any


def build_futures_causal_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # Core macroeconomic nodes
    macro = ['货币政策', '财政政策', '宏观经济', '通胀', '利率', '流动性', '汇率']
    G.add_nodes_from(macro)

    # LH-specific: supply chain
    lh_supply = ['能繁母猪存栏', '生猪存栏', '出栏量', '养殖利润', '饲料成本',
                 '豆粕价格', '玉米价格', '仔猪价格', '养殖成本', '补栏意愿']
    G.add_nodes_from(lh_supply)

    # LH-specific: demand & policy
    lh_demand = ['猪肉消费', '季节性需求', '餐饮消费', '替代品价格',
                 '收储政策', '抛储政策', '进口猪肉', '猪瘟疫情', '环保政策']
    G.add_nodes_from(lh_demand)

    # LH-specific: market
    lh_market = ['生猪现货', '生猪期货', '基差', '持仓结构', '主力动向',
                 '屠宰量', '白条批发价', '冻品库存']
    G.add_nodes_from(lh_market)

    # Event nodes (what the LLM queries)
    events = ['market_policy_change', 'supply_shock', 'demand_shift',
              'feed_cost_change', 'disease_outbreak', 'policy_intervention',
              'seasonal_effect', 'import_tariff_change']
    G.add_nodes_from(events)

    # ── Edges: macro → supply/demand ──
    G.add_edges_from([
        ('货币政策', '利率'), ('货币政策', '流动性'),
        ('财政政策', '宏观经济'), ('通胀', '货币政策'),
        ('汇率', '进口猪肉'), ('利率', '养殖成本'),
        ('流动性', '生猪期货'),
    ])

    # ── Edges: supply chain ──
    G.add_edges_from([
        ('能繁母猪存栏', '生猪存栏'), ('生猪存栏', '出栏量'),
        ('出栏量', '生猪现货'), ('补栏意愿', '仔猪价格'),
        ('仔猪价格', '养殖成本'), ('豆粕价格', '饲料成本'),
        ('玉米价格', '饲料成本'), ('饲料成本', '养殖利润'),
        ('养殖利润', '补栏意愿'), ('养殖利润', '出栏量'),
    ])

    # ── Edges: demand ──
    G.add_edges_from([
        ('猪肉消费', '生猪现货'), ('季节性需求', '猪肉消费'),
        ('餐饮消费', '猪肉消费'), ('替代品价格', '猪肉消费'),
        ('白条批发价', '猪肉消费'), ('冻品库存', '猪肉消费'),
    ])

    # ── Edges: policy → supply ──
    G.add_edges_from([
        ('收储政策', '生猪现货'), ('抛储政策', '生猪现货'),
        ('进口猪肉', '生猪现货'), ('猪瘟疫情', '出栏量'),
        ('猪瘟疫情', '能繁母猪存栏'), ('环保政策', '养殖成本'),
    ])

    # ── Edges: market structure ──
    G.add_edges_from([
        ('生猪现货', '基差'), ('生猪期货', '基差'),
        ('基差', '持仓结构'), ('持仓结构', '生猪期货'),
        ('主力动向', '生猪期货'), ('屠宰量', '生猪现货'),
    ])

    # ── Edges: event → real nodes ──
    G.add_edges_from([
        ('market_policy_change', '收储政策'),
        ('market_policy_change', '抛储政策'),
        ('market_policy_change', '进口猪肉'),
        ('supply_shock', '出栏量'),
        ('supply_shock', '能繁母猪存栏'),
        ('demand_shift', '猪肉消费'),
        ('demand_shift', '季节性需求'),
        ('feed_cost_change', '饲料成本'),
        ('feed_cost_change', '豆粕价格'),
        ('feed_cost_change', '玉米价格'),
        ('disease_outbreak', '猪瘟疫情'),
        ('disease_outbreak', '出栏量'),
        ('policy_intervention', '收储政策'),
        ('policy_intervention', '环保政策'),
        ('seasonal_effect', '季节性需求'),
        ('seasonal_effect', '猪肉消费'),
        ('import_tariff_change', '进口猪肉'),
        ('import_tariff_change', '生猪现货'),
    ])

    # Macro → LH links
    G.add_edges_from([
        ('宏观经济', '猪肉消费'), ('宏观经济', '进口猪肉'),
        ('通胀', '饲料成本'), ('利率', '生猪期货'),
    ])

    return G


def query_causal_graph(event_type: str, target: str) -> Dict[str, Any]:
    G = build_futures_causal_graph()

    # Normalize: try exact match, then fuzzy match
    node = event_type
    if node not in G.nodes:
        # Try alias matching
        aliases = {
            'lh': '生猪期货', '生猪': '生猪现货', 'pig': '生猪现货',
            'live_hog': '生猪期货', 'hog': '生猪期货',
            'policy': '收储政策', 'policy_change': 'market_policy_change',
            'feed': '饲料成本', 'corn': '玉米价格', 'soybean': '豆粕价格',
            'disease': '猪瘟疫情', 'season': '季节性需求',
            'supply': '出栏量', 'demand': '猪肉消费',
            'market_policy': 'market_policy_change',
            'macro': '宏观经济', 'fund': '主力动向',
        }
        node = aliases.get(event_type.lower().replace(' ', '_'), event_type)

    target_norm = target
    if target_norm.lower() in ('lh', '生猪', 'live_hog', 'hog'):
        target_norm = '生猪期货'

    if node not in G.nodes:
        return {
            'direction': 'NEUTRAL', 'strength': 'WEAK',
            'chain': f'{event_type} → {target} (node not found in graph)',
            'confidence': 0.1, 'net_causal_weight': 0.0,
        }

    if target_norm not in G.nodes:
        target_norm = '生猪期货'

    try:
        paths = list(nx.all_simple_paths(G, node, target_norm))
        if paths:
            shortest = min(paths, key=len)
            chain_str = ' → '.join(shortest)
            length = len(shortest) - 1

            effect_map = {1: ('STRONG', 0.85), 2: ('MODERATE', 0.70),
                          3: ('MODERATE', 0.55), 4: ('WEAK', 0.40),
                          5: ('WEAK', 0.30)}
            strength, confidence = effect_map.get(length, ('WEAK', 0.25))

            # Determine direction: if path ends in 生猪期货 and passes through
            # supply-expanding nodes → NEGATIVE (more supply = lower price)
            negative_nodes = {'出栏量', '进口猪肉', '猪瘟疫情', '抛储政策'}
            positive_nodes = {'收储政策', '猪肉消费', '季节性需求'}

            neg_hits = sum(1 for n in shortest if n in negative_nodes)
            pos_hits = sum(1 for n in shortest if n in positive_nodes)

            if neg_hits > pos_hits:
                direction = 'NEGATIVE'
            elif pos_hits > neg_hits:
                direction = 'POSITIVE'
            else:
                direction = 'POSITIVE' if '需求' in chain_str or '消费' in chain_str \
                    else ('NEGATIVE' if '存栏' in chain_str or '出栏' in chain_str else 'NEUTRAL')

            weight = confidence * (1 if direction != 'NEUTRAL' else 0.3)

            return {
                'direction': direction, 'strength': strength,
                'chain': chain_str, 'confidence': confidence,
                'net_causal_weight': weight,
            }
    except Exception:
        pass

    return {
        'direction': 'NEUTRAL', 'strength': 'WEAK',
        'chain': f'{event_type} → {target} (no path found)',
        'confidence': 0.2, 'net_causal_weight': 0.0,
    }


def do_intervention(graph: nx.DiGraph, intervention: str) -> float:
    """Apply do-calculus intervention and return effect magnitude."""
    try:
        import json
        if isinstance(intervention, str) and intervention.strip().startswith('{'):
            data = json.loads(intervention)
        else:
            data = intervention if isinstance(intervention, dict) else {}
        if not data:
            return 0.0

        total_effect = 0.0
        for var, delta in data.items():
            if var in graph.nodes:
                affected = list(nx.descendants(graph, var))
                for node in affected:
                    if '价格' in node or '期货' in node:
                        total_effect += float(delta) * 0.3
                    elif '存栏' in node or '出栏' in node:
                        total_effect += float(delta) * 0.2
        return round(total_effect, 4)
    except Exception:
        return 0.0
