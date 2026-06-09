"""Vector memory store for historical market analogues."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.schemas import MemoryCase

logger = logging.getLogger(__name__)

_db_path: Optional[Path]   = None
_collection                 = None


def init_vector_db(path: str = "./vector_db") -> None:
    global _db_path, _collection
    _db_path = Path(path)
    _db_path.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb
        client      = chromadb.PersistentClient(path=str(_db_path))
        _collection = client.get_or_create_collection("market_memories")
        logger.info("ChromaDB initialized at %s", path)
    except ImportError:
        logger.warning("chromadb not installed; using JSON fallback store")
        _collection = None


def compute_embedding(state_dict: Dict[str, Any]) -> np.ndarray:
    keys = ["ma5","ma20","rsi14","macd_hist","atr14","adx14","vol_ratio",
            "change_pct","basis_pct","oi_change_pct"]
    vec = np.array([float(state_dict.get(k, 0.0) or 0.0) for k in keys], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def add_market_snapshot(state_vector: np.ndarray, metadata: Dict[str, Any]) -> None:
    if _collection is not None:
        _collection.add(
            embeddings=[state_vector.tolist()],
            documents=[json.dumps(metadata)],
            ids=[metadata.get("date", datetime.now().isoformat())],
        )
    else:
        _json_append(state_vector, metadata)


def search_similar_markets(state_vector: np.ndarray, top_k: int = 5) -> List[MemoryCase]:
    if _collection is not None:
        try:
            results = _collection.query(
                query_embeddings=[state_vector.tolist()], n_results=top_k)
            return [
                MemoryCase(
                    date=json.loads(doc).get("date", "unknown"),
                    similarity=float(1 - dist),
                    description=json.loads(doc).get("description", ""),
                    subsequent_5d_return=float(json.loads(doc).get("subsequent_5d_return", 0.0)),
                )
                for doc, dist in zip(results["documents"][0], results["distances"][0])
            ]
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
    return _json_search(state_vector, top_k)


def _json_path() -> Path:
    base = _db_path or Path("./vector_db")
    return base / "memories.jsonl"


def _json_append(vec: np.ndarray, meta: Dict[str, Any]) -> None:
    with open(_json_path(), "a") as f:
        f.write(json.dumps({"vec": vec.tolist(), "meta": meta}) + "\n")


def _json_search(query_vec: np.ndarray, top_k: int) -> List[MemoryCase]:
    path = _json_path()
    if not path.exists():
        return _synthetic_cases()
    records = []
    with open(path) as f:
        for line in f:
            try: records.append(json.loads(line))
            except json.JSONDecodeError: continue
    if not records:
        return _synthetic_cases()
    scored = sorted(
        [(float(np.dot(query_vec, np.array(r["vec"], dtype=np.float32)) /
                (np.linalg.norm(query_vec) * np.linalg.norm(r["vec"]) + 1e-8)),
          r["meta"]) for r in records],
        key=lambda x: x[0], reverse=True,
    )
    return [MemoryCase(date=m.get("date","unknown"), similarity=s,
                       description=m.get("description",""),
                       subsequent_5d_return=float(m.get("subsequent_5d_return",0.0)))
            for s, m in scored[:top_k]]


def _synthetic_cases() -> List[MemoryCase]:
    np.random.seed(42)
    base = datetime(2023, 1, 1)
    from datetime import timedelta
    return [
        MemoryCase(
            date=(base - timedelta(days=30*(i+1))).strftime("%Y-%m-%d"),
            similarity=round(0.9 - i*0.05, 2),
            description=f"Similar market structure #{i+1}: moderate trend with volume divergence",
            subsequent_5d_return=round(float(np.random.normal(0.8, 2.5)), 2),
        ) for i in range(5)
    ]
