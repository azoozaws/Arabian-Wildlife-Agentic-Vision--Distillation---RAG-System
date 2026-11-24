"""Model Registry for versioning YOLO weights in app/models/, tracking metrics, promotion, and rollback."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.src.utils.logging_utils import get_logger

logger = get_logger("ModelRegistry")

DEFAULT_APP_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = DEFAULT_APP_DIR / "registry.json"
DEFAULT_MODELS_DIR = DEFAULT_APP_DIR / "models"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """JSON-backed model registry supporting versioned weights, evaluation logging, promotion, and rollback."""

    def __init__(
        self,
        registry_path: Path | str = DEFAULT_REGISTRY_PATH,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load_or_init()

    def _default_schema(self) -> Dict[str, Any]:
        return {
            "active_champion": None,
            "previous_champion": None,
            "models_dir": str(self.models_dir.resolve()),
            "updated_at": _utc_now_iso(),
            "models": [],
            "history": [],
        }

    def _normalize_entry_paths(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-resolve cross-platform weights paths (e.g., Windows registry.json uploaded to Colab Linux)."""
        raw_wp = str(entry.get("weights_path") or "")
        if raw_wp and Path(raw_wp).exists():
            return entry
        # Try matching filename or <model_id>.pt inside current models_dir
        candidates = [self.models_dir / f"{entry.get('model_id', '')}.pt"]
        if raw_wp:
            fname = raw_wp.replace("\\", "/").split("/")[-1]
            if fname:
                candidates.insert(0, self.models_dir / fname)
        for cand in candidates:
            if cand.exists():
                entry["weights_path"] = str(cand.resolve())
                break
        return entry

    def _load_or_init(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            raw = self.registry_path.read_text(encoding="utf-8").strip()
            if raw:
                data = json.loads(raw)
                data.setdefault("models", [])
                data.setdefault("history", [])
                data["models_dir"] = str(self.models_dir.resolve())
                for entry in data["models"]:
                    self._normalize_entry_paths(entry)
                return data
        data = self._default_schema()
        self._save(data)
        return data

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        payload = data if data is not None else self._data
        payload["updated_at"] = _utc_now_iso()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-style write via temp file to prevent corrupting registry.json on interruption
        tmp_path = self.registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def reload(self) -> Dict[str, Any]:
        self._data = self._load_or_init()
        return self._data

    def _find_entry(self, model_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._data["models"]:
            if entry["model_id"] == model_id:
                return entry
        return None

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._find_entry(model_id)

    def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        models = list(self._data["models"])
        if status:
            return [m for m in models if m.get("status") == status]
        return models

    def get_active_champion(self) -> Optional[Dict[str, Any]]:
        active_id = self._data.get("active_champion")
        if not active_id:
            return None
        return self._find_entry(active_id)

    def get_previous_champion(self) -> Optional[Dict[str, Any]]:
        prev_id = self._data.get("previous_champion")
        if not prev_id:
            return None
        return self._find_entry(prev_id)

    def register_trained_model(
        self,
        source_weights_path: Path | str,
        model_id: str,
        version_tag: str,
        parent_model_id: Optional[str] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "candidate",
        copy_to_models_dir: bool = True,
    ) -> Dict[str, Any]:
        """Store trained weights in models_dir and register or update the model entry in registry.json."""
        src_path = Path(source_weights_path)
        dst_path = self.models_dir / f"{model_id}.pt"

        if copy_to_models_dir:
            if not src_path.exists():
                raise FileNotFoundError(f"Trained weights not found at: {src_path}")
            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
            stored_weights_path = str(dst_path.resolve())
        else:
            stored_weights_path = str(src_path.resolve())

        now = _utc_now_iso()
        existing = self._find_entry(model_id)
        entry: Dict[str, Any] = {
            "model_id": model_id,
            "version_tag": version_tag,
            "weights_path": stored_weights_path,
            "parent_model_id": parent_model_id or self._data.get("active_champion"),
            "status": status,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
            "promoted_at": existing.get("promoted_at") if existing else None,
            "metrics": metrics if metrics is not None else (existing.get("metrics") if existing else None),
            "gate_decision": existing.get("gate_decision", "pending") if existing else "pending",
            "gate_reason": existing.get("gate_reason", "") if existing else "",
            "training_metadata": training_metadata or (existing.get("training_metadata", {}) if existing else {}),
        }

        if existing is not None:
            existing.update(entry)
        else:
            self._data["models"].append(entry)

        self._data["history"].append(
            {
                "timestamp": now,
                "action": "register_model",
                "model_id": model_id,
                "status": status,
                "weights_path": stored_weights_path,
            }
        )
        self._save()
        logger.info("Registered model '%s' (status=%s) -> %s", model_id, status, stored_weights_path)
        return entry

    def record_evaluation(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        gate_decision: str = "pending",
        gate_reason: str = "",
        champion_points: Optional[int] = None,
        challenger_points: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach evaluation MetricsSnapshot and optional gate outcome to a registered model."""
        entry = self._find_entry(model_id)
        if entry is None:
            raise KeyError(f"Model '{model_id}' is not registered in {self.registry_path}.")

        now = _utc_now_iso()
        entry["metrics"] = dict(metrics)
        entry["gate_decision"] = gate_decision
        entry["gate_reason"] = gate_reason
        entry["updated_at"] = now
        if champion_points is not None or challenger_points is not None:
            entry["gate_scorecard"] = {
                "champion_points": champion_points,
                "challenger_points": challenger_points,
            }

        self._data["history"].append(
            {
                "timestamp": now,
                "action": "record_evaluation",
                "model_id": model_id,
                "gate_decision": gate_decision,
                "metrics": dict(metrics),
            }
        )
        self._save()
        return entry

    def promote_model(
        self,
        model_id: str,
        metrics: Optional[Dict[str, Any]] = None,
        gate_reason: str = "",
        engineer_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote a registered model to active Champion and archive the previous Champion for rollback."""
        entry = self._find_entry(model_id)
        if entry is None:
            raise KeyError(f"Cannot promote unknown model '{model_id}'.")

        weights_file = Path(entry["weights_path"])
        if not weights_file.exists():
            raise FileNotFoundError(f"Cannot promote '{model_id}'; weights missing at {weights_file}.")

        now = _utc_now_iso()
        old_champ_id = self._data.get("active_champion")
        if old_champ_id and old_champ_id != model_id:
            old_entry = self._find_entry(old_champ_id)
            if old_entry is not None:
                old_entry["status"] = "archived"
                old_entry["updated_at"] = now
            self._data["previous_champion"] = old_champ_id

        entry["status"] = "champion"
        entry["promoted_at"] = now
        entry["updated_at"] = now
        if metrics is not None:
            entry["metrics"] = dict(metrics)
        entry["gate_decision"] = "promote"
        if gate_reason:
            entry["gate_reason"] = gate_reason
        if engineer_override:
            entry["engineer_override"] = engineer_override

        self._data["active_champion"] = model_id
        self._data["history"].append(
            {
                "timestamp": now,
                "action": "promote",
                "model_id": model_id,
                "previous_champion": old_champ_id,
                "gate_reason": gate_reason,
                "engineer_override": engineer_override,
            }
        )
        self._save()
        logger.info("Promoted '%s' to active Champion (previous: %s).", model_id, old_champ_id)
        return entry

    def reject_model(
        self,
        model_id: str,
        gate_reason: str = "",
        engineer_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a candidate model as rejected while leaving the active Champion untouched."""
        entry = self._find_entry(model_id)
        if entry is None:
            raise KeyError(f"Cannot reject unknown model '{model_id}'.")

        now = _utc_now_iso()
        entry["status"] = "rejected"
        entry["gate_decision"] = "keep_champion"
        entry["updated_at"] = now
        if gate_reason:
            entry["gate_reason"] = gate_reason
        if engineer_override:
            entry["engineer_override"] = engineer_override

        self._data["history"].append(
            {
                "timestamp": now,
                "action": "reject",
                "model_id": model_id,
                "gate_reason": gate_reason,
                "engineer_override": engineer_override,
            }
        )
        self._save()
        logger.info("Rejected candidate '%s'. Active Champion remains '%s'.", model_id, self._data.get("active_champion"))
        return entry

    def rollback(
        self,
        target_model_id: Optional[str] = None,
        reason: str = "Manual or automated rollback due to new champion failure",
    ) -> Dict[str, Any]:
        """Roll back active Champion to previous_champion (or a specific archived model_id)."""
        current_id = self._data.get("active_champion")
        resolved_target_id = target_model_id or self._data.get("previous_champion")

        if not resolved_target_id:
            # Fallback to the most recently updated archived model
            archived = [m for m in self._data["models"] if m.get("status") == "archived"]
            if archived:
                resolved_target_id = archived[-1]["model_id"]

        if not resolved_target_id:
            raise ValueError("No previous or archived model available for rollback.")

        target_entry = self._find_entry(resolved_target_id)
        if target_entry is None:
            raise KeyError(f"Rollback target model '{resolved_target_id}' not found in registry.")

        target_weights = Path(target_entry["weights_path"])
        if not target_weights.exists():
            raise FileNotFoundError(
                f"Cannot roll back to '{resolved_target_id}'; weights missing at {target_weights}."
            )

        now = _utc_now_iso()
        if current_id and current_id != resolved_target_id:
            curr_entry = self._find_entry(current_id)
            if curr_entry is not None:
                curr_entry["status"] = "rolled_back"
                curr_entry["updated_at"] = now
                curr_entry["rollback_reason"] = reason

        target_entry["status"] = "champion"
        target_entry["updated_at"] = now
        self._data["previous_champion"] = current_id if current_id != resolved_target_id else self._data.get("previous_champion")
        self._data["active_champion"] = resolved_target_id

        self._data["history"].append(
            {
                "timestamp": now,
                "action": "rollback",
                "from_model_id": current_id,
                "to_model_id": resolved_target_id,
                "reason": reason,
            }
        )
        self._save()
        logger.info("Rolled back Champion from '%s' -> '%s'.", current_id, resolved_target_id)
        return target_entry
