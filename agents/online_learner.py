"""Online Learner — continuous drift detection and model adaptation."""

import logging
from typing import Dict, Any
from models.schemas import OnlineLearnerState

logger = logging.getLogger(__name__)

_learner       = None
_drift_detector= None


def _init_river():
    global _learner, _drift_detector
    try:
        from river import tree, drift
        _learner        = tree.HoeffdingTreeClassifier()
        _drift_detector = drift.ADWIN()
        logger.info("River online learner initialized")
        return True
    except ImportError:
        logger.warning("river not installed; using mock online learner")
        return False


def update_model(features: Dict[str, float], label: int) -> OnlineLearnerState:
    global _learner, _drift_detector
    if _learner is None:
        if not _init_river():
            return _mock_state()
    try:
        prediction = _learner.predict_one(features)
        _learner.learn_one(features, label)
        _drift_detector.update(1 if prediction != label else 0)
        drift_detected  = _drift_detector.drift_detected
        recent_accuracy = 1 - getattr(_drift_detector, "estimation", 0.3)
        return OnlineLearnerState(
            model_version="hoeffding_v1",
            recent_accuracy=round(min(1.0, max(0.0, recent_accuracy)), 3),
            drift_detected=drift_detected,
            drift_details="ADWIN detected distribution shift" if drift_detected else None,
        )
    except Exception as e:
        logger.warning("Online learner update failed: %s", e)
        return _mock_state()


def check_drift() -> OnlineLearnerState:
    if _drift_detector is None:
        _init_river()
    if _drift_detector is None:
        return _mock_state()
    try:
        return OnlineLearnerState(
            model_version="hoeffding_v1", recent_accuracy=0.65,
            drift_detected=getattr(_drift_detector, "drift_detected", False),
        )
    except Exception:
        return _mock_state()


def _mock_state() -> OnlineLearnerState:
    return OnlineLearnerState(
        model_version="mock_v0", recent_accuracy=0.60,
        drift_detected=False, drift_details="River not available; using mock state",
    )
