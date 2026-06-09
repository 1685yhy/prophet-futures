from models.schemas import ABMResult
from tools.abm_engine import run_abm


def run_abm_simulation(order_book_snapshot: dict, n_agents: int = 500, steps: int = 100) -> ABMResult:
    return run_abm(order_book_snapshot, n_agents, steps)