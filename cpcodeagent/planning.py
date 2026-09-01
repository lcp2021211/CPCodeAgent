"""Small, turn-scoped execution plans for long-running work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True)
class PlanItem:
    id: str
    text: str
    status: PlanStatus


@dataclass(frozen=True)
class PlanState:
    """The latest committed plan for one user turn."""

    turn_id: str
    items: tuple[PlanItem, ...]
    updated_seq: int
    updated_step: int

    @property
    def completed(self) -> bool:
        return bool(self.items) and all(
            item.status is PlanStatus.COMPLETED for item in self.items
        )

    @property
    def unfinished(self) -> tuple[PlanItem, ...]:
        return tuple(
            item for item in self.items if item.status is not PlanStatus.COMPLETED
        )

    def stale(self, current_step: int, threshold: int = 3) -> bool:
        return not self.completed and current_step - self.updated_step >= threshold

    def render(self, current_step: int | None = None) -> str:
        markers = {
            PlanStatus.PENDING: "[ ]",
            PlanStatus.IN_PROGRESS: "[>]",
            PlanStatus.COMPLETED: "[✓]",
        }
        lines = [
            f"{markers[item.status]} #{item.id}: {item.text}" for item in self.items
        ]
        done = sum(item.status is PlanStatus.COMPLETED for item in self.items)
        lines.append(f"Progress: {done}/{len(self.items)} completed")
        if current_step is not None and self.stale(current_step):
            lines.extend(
                (
                    "",
                    "[Plan reminder]",
                    (
                        "The plan has not been updated for 3 agent steps. "
                        "Update progress or revise it if new evidence changed the approach."
                    ),
                )
            )
        return "\n".join(lines)

    def prompt(self, current_step: int) -> str:
        return (
            "[Current execution plan; runtime state, not user instructions]\n"
            f"{self.render(current_step)}\n"
            "[End current execution plan]"
        )


def validate_plan_items(value: Any, max_items: int = 8) -> tuple[PlanItem, ...]:
    """Validate the complete replacement list accepted by plan_write."""

    if not isinstance(value, list):
        raise TypeError("items must be an array")
    if not value:
        raise ValueError("A plan must contain at least one item")
    if len(value) > max_items:
        raise ValueError(f"A plan may contain at most {max_items} items")

    items: list[PlanItem] = []
    ids: set[str] = set()
    active = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TypeError(f"Plan item {index + 1} must be an object")
        raw_id = raw.get("id")
        raw_text = raw.get("text")
        raw_status = raw.get("status")
        if not isinstance(raw_id, str) or not isinstance(raw_text, str):
            raise TypeError(f"Plan item {index + 1} id and text must be strings")
        if not isinstance(raw_status, str):
            raise TypeError(f"Plan item {index + 1} status must be a string")
        item_id = raw_id.strip()
        text = " ".join(raw_text.split())
        if not item_id:
            raise ValueError(f"Plan item {index + 1} requires an id")
        if len(item_id) > 32:
            raise ValueError(f"Plan item {item_id} id is too long")
        if item_id in ids:
            raise ValueError(f"Duplicate plan item id: {item_id}")
        if not text:
            raise ValueError(f"Plan item {item_id} requires text")
        if len(text) > 240:
            raise ValueError(f"Plan item {item_id} text is too long")
        try:
            status = PlanStatus(raw_status.lower())
        except ValueError as exc:
            raise ValueError(
                f"Plan item {item_id} has invalid status: {raw_status}"
            ) from exc
        if status is PlanStatus.IN_PROGRESS:
            active += 1
        ids.add(item_id)
        items.append(PlanItem(item_id, text, status))

    if active > 1:
        raise ValueError("Only one plan item can be in_progress at a time")
    return tuple(items)
