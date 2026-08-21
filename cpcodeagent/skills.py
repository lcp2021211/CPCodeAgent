"""Filesystem skills with small catalogs and on-demand instruction loading."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    required_tools: tuple[str, ...]
    directory: Path
    digest: str


class SkillRegistry:
    """Discovers SKILL.md files. Earlier roots have higher precedence."""

    def __init__(self, roots: Iterable[str | Path] = ()):
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self._skills: dict[str, Skill] | None = None

    def refresh(self) -> None:
        self._skills = self._discover()

    def catalog(self) -> tuple[dict[str, str], ...]:
        skills = self._get_all()
        return tuple(
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.digest[:12],
            }
            for skill in sorted(skills.values(), key=lambda item: item.name)
        )

    def get(self, name: str) -> Skill:
        try:
            return self._get_all()[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def read_resource(self, name: str, relative_path: str) -> str:
        skill = self.get(name)
        target = (skill.directory / relative_path).resolve()
        try:
            target.relative_to(skill.directory)
        except ValueError as exc:
            raise ValueError("Skill resource path escapes its directory") from exc
        if not target.is_file():
            raise FileNotFoundError(f"Skill resource does not exist: {relative_path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def _get_all(self) -> dict[str, Skill]:
        if self._skills is None:
            self._skills = self._discover()
        return self._skills

    def _discover(self) -> dict[str, Skill]:
        discovered: dict[str, Skill] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            candidates = [root] if (root / "SKILL.md").is_file() else sorted(root.iterdir())
            for directory in candidates:
                skill_file = directory / "SKILL.md"
                if not directory.is_dir() or not skill_file.is_file():
                    continue
                skill = _load_skill(skill_file)
                discovered.setdefault(skill.name, skill)
        return discovered


def _load_skill(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _frontmatter(raw)
    name = metadata.get("name", path.parent.name).strip()
    description = metadata.get("description", "").strip()
    if not name or not description:
        raise ValueError(f"Skill {path} must define a name and description")
    required = _parse_list(metadata.get("requires-tools", metadata.get("required-tools", "")))
    return Skill(
        name=name,
        description=description,
        instructions=body.strip(),
        required_tools=required,
        directory=path.parent.resolve(),
        digest=hashlib.sha256(raw.encode()).hexdigest(),
    )


def _frontmatter(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise ValueError("Unclosed SKILL.md frontmatter") from exc
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid SKILL.md frontmatter line: {line}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, "\n".join(lines[end + 1 :])


def _parse_list(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            # YAML permits unquoted scalar lists such as [read_file, edit_file].
            parsed = [item.strip() for item in raw[1:-1].split(",")]
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("requires-tools must be a list")
        return tuple(
            str(item).strip().strip("'\"") for item in parsed if str(item).strip()
        )
    return tuple(item.strip() for item in raw.split(",") if item.strip())
