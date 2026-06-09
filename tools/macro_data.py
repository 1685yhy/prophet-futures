import numpy as np
from typing import List, Dict


def get_sector_performance(sector: str = None) -> Dict:
    sectors = {
        '黑色系': ['螺纹钢', '铁矿石', '焦炭', '焦煤'],
        '有色系': ['铜', '铝', '锌', '镍'],
        '能源化工': ['原油', '甲醇', 'PTA', '沥青'],
        '农产品': ['大豆', '玉米', '棉花', '白糖']
    }
    
    result = {}
    if sector and sector in sectors:
        for item in sectors[sector]:
            result[item] = np.random.normal(0, 2)
    else:
        for sector_name, items in sectors.items():
            result[sector_name] = {item: np.random.normal(0, 2) for item in items}
    
    return result


def get_market_index() -> Dict:
    return {
        '上证指数': 3200 + np.random.normal(0, 50),
        '沪深300': 4100 + np.random.normal(0, 80),
        '创业板指': 2300 + np.random.normal(0, 40),
        '上证50': 2500 + np.random.normal(0, 30),
        '波动率指数': 15 + np.random.normal(0, 3)
    }


def get_global_futures() -> Dict:
    return {
        '外盘原油': 75 + np.random.normal(0, 3),
        'COMEX黄金': 2000 + np.random.normal(0, 30),
        'LME铜': 8500 + np.random.normal(0, 100),
        'CBOT大豆': 1150 + np.random.normal(0, 20),
        '美元指数': 102 + np.random.normal(0, 1),
        '美债收益率': 4.2 + np.random.normal(0, 0.1)
    }


def get_news_sentiment(keywords: List[str] = None) -> List[Dict]:
    news_templates = [
        {'title': '央行宣布下调存款准备金率', 'content': '央行决定下调金融机构存款准备金率0.25个百分点', 'sentiment': 0.7, 'category': '货币政策'},
        {'title': '发改委发布钢铁行业产能调控政策', 'content': '发改委表示将继续推进钢铁行业去产能工作', 'sentiment': -0.3, 'category': '行业政策'},
        {'title': '美国CPI数据低于预期', 'content': '美国公布的CPI数据低于市场预期，市场预期美联储将暂停加息', 'sentiment': 0.5, 'category': '宏观经济'},
        {'title': '地缘政治紧张局势升级', 'content': '中东地区地缘政治紧张局势进一步升级', 'sentiment': -0.6, 'category': '国际形势'},
        {'title': '国内PMI数据回暖', 'content': '中国制造业PMI数据回升至荣枯线以上', 'sentiment': 0.4, 'category': '宏观经济'}
    ]
    
    if keywords:
        filtered = []
        for news in news_templates:
            if any(keyword in news['title'] or keyword in news['content'] for keyword in keywords):
                filtered.append(news)
        return filtered
    
    return news_templates