"""GhostGUI project folder metadata and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from uuid import uuid4

from application.paths import PROJECT_ROOT, atomic_text_writer


PROJECT_FILENAME = "ghostgui_project.json"
PROJECT_SCHEMA_VERSION = 2
RECENT_PROJECTS_FILENAME = "recent_projects.json"
PROJECTS_DIR_ENV = "GHOSTGUI_PROJECTS_DIR"
PROJECTS_FOLDER_NAME = "projects"
MAX_RECENT_PROJECTS = 10
TRANSACTIONS_DIRNAME = ".ghostgui-transactions"
TRANSACTION_JOURNAL_FILENAME = "journal.json"
TRANSACTION_SCHEMA_VERSION = 1
TRANSACTION_STATUSES = frozenset(
    {"prepared", "replacing", "committed", "rolled_back"}
)

DEFAULT_PROJECT_FILES = {
    "target_trajectory": "data/target_frames.json",
    "qpos_timeline": "data/qpos_timeline.npz",
    "workspace": "workspace/workspace.json",
    "last_snapshot": "snapshots/last_workspace.png",
    "session_log": "metadata/session_log.jsonl",
}

DEFAULT_AUTOSAVE_FILES = {
    "manifest": "autosave/autosave_manifest.json",
    "target_trajectory": "autosave/target_frames.autosave.json",
    "qpos_timeline": "autosave/qpos_timeline.autosave.npz",
    "workspace": "autosave/workspace.autosave.json",
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path, data):
    path = Path(path)
    with atomic_text_writer(path) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class ProjectFormatError(ValueError):
    """Raised when project metadata violates a schema or path contract."""


def _migrate_v0_to_v1(metadata):
    migrated = deepcopy(metadata)
    now = utc_now_iso()
    model_key = str(
        migrated.pop("model_key", None)
        or migrated.get("robot", {}).get("model_key")
        or "g1"
    )
    migrated.setdefault("project_name", "GhostGUI Project")
    migrated.setdefault("created_at", now)
    migrated.setdefault("modified_at", migrated["created_at"])
    migrated.setdefault(
        "application",
        {"name": "GhostGUI", "project_format": "ghostgui.project.v1"},
    )
    migrated.setdefault(
        "robot",
        {"model_key": model_key, "model_name": model_key},
    )
    migrated.setdefault("files", dict(DEFAULT_PROJECT_FILES))
    migrated["schema_version"] = 1
    return migrated


def _migrate_v1_to_v2(metadata):
    migrated = deepcopy(metadata)
    migrated.setdefault("autosave_files", dict(DEFAULT_AUTOSAVE_FILES))
    application = migrated.setdefault("application", {})
    application.setdefault("name", "GhostGUI")
    application["project_format"] = "ghostgui.project.v2"
    migrated.setdefault(
        "persistence",
        {"transaction_format": "ghostgui.transaction.v1"},
    )
    migrated["schema_version"] = 2
    return migrated


PROJECT_MIGRATIONS = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
}


def migrate_project_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ProjectFormatError("project metadata must be a JSON object")
    migrated = deepcopy(metadata)
    try:
        version = int(migrated.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError("project schema_version must be an integer") from exc
    if version < 0:
        raise ProjectFormatError("project schema_version cannot be negative")
    if version > PROJECT_SCHEMA_VERSION:
        raise ProjectFormatError(
            f"Project schema {version} is newer than this GhostGUI "
            f"(supports through {PROJECT_SCHEMA_VERSION})"
        )
    while version < PROJECT_SCHEMA_VERSION:
        migration = PROJECT_MIGRATIONS.get(version)
        if migration is None:
            raise ProjectFormatError(
                f"No migration is available from project schema {version}"
            )
        migrated = migration(migrated)
        version = int(migrated.get("schema_version", -1))
    return migrated


def safe_project_path(root_dir, relative_path, *, label="project file"):
    root = Path(root_dir).expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ProjectFormatError(f"{label} must be relative: {relative}")
    if not relative.parts or relative == Path(".") or ".." in relative.parts:
        raise ProjectFormatError(f"{label} is not a safe relative path: {relative}")
    if relative.parts[0] == TRANSACTIONS_DIRNAME:
        raise ProjectFormatError(
            f"{label} cannot use the internal transaction directory"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ProjectFormatError(f"{label} escapes the project folder: {relative}")
    return resolved


def _relative_project_path(root_dir, path):
    root = Path(root_dir).expanduser().resolve()
    path = Path(path).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProjectFormatError(
            f"transaction destination is outside the project: {path}"
        ) from exc
    safe_project_path(root, relative, label="transaction destination")
    return relative


def _safe_transaction_artifact_path(
    transaction_root,
    artifact_root,
    relative_path,
):
    transaction_root = Path(transaction_root).resolve()
    artifact_root = Path(artifact_root)
    candidate = artifact_root / Path(relative_path)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(transaction_root):
        raise ProjectFormatError(
            f"transaction artifact escapes its directory: {candidate}"
        )
    return candidate


def _fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    try:
        descriptor = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_timeline_save(timeline, destination):
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        timeline.save_npz(temporary_path)
        _fsync_file(temporary_path)
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class ProjectSaveTransaction:
    """Stage a multi-file save and roll it back after errors or interruption."""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.identifier = uuid4().hex
        self.transactions_root = self.root_dir / TRANSACTIONS_DIRNAME
        if self.transactions_root.is_symlink():
            raise ProjectFormatError(
                "project transaction directory cannot be a symbolic link"
            )
        self.transaction_root = self.transactions_root / self.identifier
        self.staging_root = self.transaction_root / "staging"
        self.backup_root = self.transaction_root / "backup"
        self.journal_path = (
            self.transaction_root / TRANSACTION_JOURNAL_FILENAME
        )
        self._operations = []
        self._committed = False

    def _stage_path(self, destination):
        relative = _relative_project_path(self.root_dir, destination)
        if relative in self._operations:
            raise ValueError(f"destination staged twice: {relative}")
        self._operations.append(relative)
        path = _safe_transaction_artifact_path(
            self.transaction_root,
            self.staging_root,
            relative,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def stage_json(self, destination, payload):
        staging_path = self._stage_path(destination)
        _write_json(staging_path, payload)

    def stage_timeline(self, destination, timeline):
        staging_path = self._stage_path(destination)
        timeline.save_npz(staging_path)
        _fsync_file(staging_path)

    def commit(self, *, replace=os.replace):
        if self._committed:
            raise RuntimeError("transaction was already committed")
        if not self._operations:
            raise RuntimeError("transaction has no staged files")

        self.backup_root.mkdir(parents=True, exist_ok=True)
        absent = []
        for relative in self._operations:
            destination = safe_project_path(
                self.root_dir,
                relative,
                label="transaction destination",
            )
            backup = _safe_transaction_artifact_path(
                self.transaction_root,
                self.backup_root,
                relative,
            )
            if destination.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
                _fsync_file(backup)
            else:
                absent.append(relative.as_posix())

        journal = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "transaction_id": self.identifier,
            "status": "prepared",
            "operations": [path.as_posix() for path in self._operations],
            "absent_before": absent,
        }
        _write_json(self.journal_path, journal)
        _fsync_directory(self.transaction_root)

        try:
            journal["status"] = "replacing"
            _write_json(self.journal_path, journal)
            for relative in self._operations:
                source = _safe_transaction_artifact_path(
                    self.transaction_root,
                    self.staging_root,
                    relative,
                )
                destination = safe_project_path(
                    self.root_dir,
                    relative,
                    label="transaction destination",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                replace(source, destination)
                _fsync_directory(destination.parent)
            journal["status"] = "committed"
            _write_json(self.journal_path, journal)
            _fsync_directory(self.transaction_root)
            self._committed = True
        except Exception:
            self._rollback(journal)
            raise
        finally:
            if self._committed or journal.get("status") == "rolled_back":
                self._cleanup()

    def _rollback(self, journal):
        absent = set(journal.get("absent_before", []))
        for raw_relative in reversed(journal.get("operations", [])):
            relative = Path(raw_relative)
            destination = safe_project_path(
                self.root_dir,
                relative,
                label="transaction rollback destination",
            )
            backup = _safe_transaction_artifact_path(
                self.transaction_root,
                self.backup_root,
                relative,
            )
            if backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
                _fsync_directory(destination.parent)
            elif relative.as_posix() in absent:
                destination.unlink(missing_ok=True)
                _fsync_directory(destination.parent)
        journal["status"] = "rolled_back"
        _write_json(self.journal_path, journal)

    def _cleanup(self):
        shutil.rmtree(self.transaction_root, ignore_errors=True)
        try:
            self.transactions_root.rmdir()
        except OSError:
            pass


def _validate_transaction_journal(root_dir, transaction_root, journal):
    if not isinstance(journal, dict):
        raise ProjectFormatError("transaction journal must be a JSON object")
    if journal.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise ProjectFormatError(
            "unsupported transaction journal schema: "
            f"{journal.get('schema_version')!r}"
        )
    status = journal.get("status")
    if status not in TRANSACTION_STATUSES:
        raise ProjectFormatError(
            f"invalid transaction journal status: {status!r}"
        )
    identifier = journal.get("transaction_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ProjectFormatError("transaction journal has no transaction_id")
    if identifier != transaction_root.name:
        raise ProjectFormatError(
            "transaction journal identifier does not match its directory"
        )
    operations = journal.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ProjectFormatError("transaction journal has no operations")
    normalized = []
    seen = set()
    for value in operations:
        if not isinstance(value, str):
            raise ProjectFormatError(
                "transaction journal operation must be a relative string"
            )
        relative = Path(value)
        safe_project_path(
            root_dir,
            relative,
            label="transaction journal operation",
        )
        key = relative.as_posix()
        if key in seen:
            raise ProjectFormatError(
                f"transaction journal repeats operation: {key}"
            )
        seen.add(key)
        normalized.append(relative)
    absent = journal.get("absent_before")
    if not isinstance(absent, list) or any(
        not isinstance(value, str) or value not in seen
        for value in absent
    ):
        raise ProjectFormatError(
            "transaction absent_before must reference journal operations"
        )
    return normalized


def recover_project_transactions(root_dir):
    root_dir = Path(root_dir).expanduser().resolve()
    transactions_root = root_dir / TRANSACTIONS_DIRNAME
    if transactions_root.is_symlink():
        raise ProjectFormatError(
            "project transaction directory cannot be a symbolic link"
        )
    if not transactions_root.is_dir():
        return []
    recovered = []
    for transaction_root in sorted(transactions_root.iterdir()):
        if not transaction_root.is_dir():
            continue
        if transaction_root.is_symlink():
            raise ProjectFormatError(
                "project transaction cannot be a symbolic link: "
                f"{transaction_root.name}"
            )
        journal_path = transaction_root / TRANSACTION_JOURNAL_FILENAME
        try:
            journal = _read_json(journal_path)
        except FileNotFoundError:
            # No destination is replaced before a valid journal is present.
            shutil.rmtree(transaction_root, ignore_errors=True)
            continue
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectFormatError(
                f"Cannot safely recover transaction {transaction_root.name}: "
                f"invalid journal ({exc})"
            ) from exc
        operations = _validate_transaction_journal(
            root_dir,
            transaction_root,
            journal,
        )
        transaction = ProjectSaveTransaction.__new__(ProjectSaveTransaction)
        transaction.root_dir = root_dir
        transaction.identifier = journal["transaction_id"]
        transaction.transactions_root = transactions_root
        transaction.transaction_root = transaction_root
        transaction.staging_root = transaction_root / "staging"
        transaction.backup_root = transaction_root / "backup"
        transaction.journal_path = journal_path
        transaction._operations = operations
        transaction._committed = journal.get("status") == "committed"
        if transaction._committed:
            transaction._cleanup()
            continue
        transaction._rollback(journal)
        recovered.append(transaction.identifier)
        transaction._cleanup()
    try:
        transactions_root.rmdir()
    except OSError:
        pass
    return recovered


def sanitized_project_folder_name(project_name):
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", project_name.strip())
    name = re.sub(r"\s+", "_", name).strip("._-")
    if not name:
        name = "ghostgui_project"
    if not name.endswith(".ghostgui"):
        name = f"{name}.ghostgui"
    return name


def project_root_from_name(parent_dir, project_name):
    return Path(parent_dir).expanduser() / sanitized_project_folder_name(project_name)


def ghostgui_projects_dir():
    override = os.environ.get(PROJECTS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / PROJECTS_FOLDER_NAME


def default_project_root_from_name(project_name):
    return project_root_from_name(ghostgui_projects_dir(), project_name)


def available_project_root_from_name(parent_dir, project_name):
    base = project_root_from_name(parent_dir, project_name)
    if not base.exists():
        return base
    suffix = ".ghostgui"
    stem = base.name[:-len(suffix)] if base.name.endswith(suffix) else base.name
    for index in range(2, 1000):
        candidate = base.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available project folder for {base}")


def available_default_project_root_from_name(project_name):
    return available_project_root_from_name(ghostgui_projects_dir(), project_name)


def ghostgui_config_dir():
    override = os.environ.get("GHOSTGUI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GhostGUI"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "GhostGUI"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "ghostgui"
    return Path.home() / ".config" / "ghostgui"


def recent_projects_path(config_dir=None):
    root = Path(config_dir).expanduser() if config_dir else ghostgui_config_dir()
    return root / RECENT_PROJECTS_FILENAME


def _project_root_from_recent_path(path):
    project_path = Path(path).expanduser()
    if project_path.name == PROJECT_FILENAME:
        return project_path.parent
    return project_path


def _recent_project_key(path):
    return str(_project_root_from_recent_path(path).resolve())


def _is_project_root(path):
    return (_project_root_from_recent_path(path) / PROJECT_FILENAME).exists()


def _recent_project_entry(project):
    return {
        "path": str(project.root_dir),
        "project_name": project.project_name,
        "model_key": project.model_key,
        "last_opened_at": utc_now_iso(),
    }


def _clean_recent_project_entries(entries, limit=MAX_RECENT_PROJECTS):
    cleaned = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not raw_path:
            continue
        project_root = _project_root_from_recent_path(raw_path)
        if not _is_project_root(project_root):
            continue
        key = _recent_project_key(project_root)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "path": key,
                "project_name": str(
                    entry.get("project_name") or project_root.stem
                ),
                "model_key": str(entry.get("model_key") or ""),
                "last_opened_at": str(entry.get("last_opened_at") or ""),
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def load_recent_projects(config_dir=None, limit=MAX_RECENT_PROJECTS):
    path = recent_projects_path(config_dir)
    try:
        payload = _read_json(path)
    except FileNotFoundError:
        return []
    except (OSError, TypeError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        entries = payload.get("projects", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        return []
    return _clean_recent_project_entries(entries, limit=limit)


def save_recent_projects(entries, config_dir=None, limit=MAX_RECENT_PROJECTS):
    cleaned = _clean_recent_project_entries(entries, limit=limit)
    _write_json(
        recent_projects_path(config_dir),
        {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "projects": cleaned,
        },
    )
    return cleaned


def remember_recent_project(project, config_dir=None, limit=MAX_RECENT_PROJECTS):
    entry = _recent_project_entry(project)
    entry_key = _recent_project_key(entry["path"])
    entries = [entry]
    entries.extend(
        recent
        for recent in load_recent_projects(config_dir, limit=limit)
        if _recent_project_key(recent["path"]) != entry_key
    )
    return save_recent_projects(entries, config_dir=config_dir, limit=limit)


def forget_recent_project(path, config_dir=None, limit=MAX_RECENT_PROJECTS):
    remove_key = _recent_project_key(path)
    entries = [
        entry
        for entry in load_recent_projects(config_dir, limit=limit)
        if _recent_project_key(entry["path"]) != remove_key
    ]
    return save_recent_projects(entries, config_dir=config_dir, limit=limit)


def project_preview_from_project(project, last_opened_at=""):
    robot = project.metadata.get("robot", {})
    snapshot_path = project.paths.last_snapshot
    return {
        "path": str(project.root_dir),
        "project_name": project.project_name,
        "model_key": project.model_key,
        "model_name": str(robot.get("model_name") or project.model_key),
        "modified_at": str(project.metadata.get("modified_at") or ""),
        "last_opened_at": str(last_opened_at or ""),
        "snapshot_path": str(snapshot_path) if snapshot_path.exists() else "",
    }


def project_preview_from_recent_entry(entry):
    project = GhostGUIProject.open(entry["path"])
    return project_preview_from_project(
        project,
        last_opened_at=entry.get("last_opened_at"),
    )


def project_roots_in_dir(projects_dir=None):
    root = Path(projects_dir).expanduser() if projects_dir else ghostgui_projects_dir()
    if not root.exists():
        return []
    roots = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / PROJECT_FILENAME).exists()
    ]
    return sorted(
        roots,
        key=lambda path: (path / PROJECT_FILENAME).stat().st_mtime,
        reverse=True,
    )


def load_recent_project_previews(config_dir=None, limit=MAX_RECENT_PROJECTS):
    previews = []
    for entry in load_recent_projects(config_dir, limit=limit):
        try:
            previews.append(project_preview_from_recent_entry(entry))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return previews


def load_project_browser_previews(
    config_dir=None,
    projects_dir=None,
    limit=MAX_RECENT_PROJECTS,
):
    previews = []
    seen = set()
    for preview in load_recent_project_previews(config_dir, limit=limit):
        key = _recent_project_key(preview["path"])
        if key in seen:
            continue
        seen.add(key)
        previews.append(preview)
    for root in project_roots_in_dir(projects_dir):
        key = _recent_project_key(root)
        if key in seen:
            continue
        try:
            previews.append(project_preview_from_project(GhostGUIProject.open(root)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        seen.add(key)
    return previews


@dataclass(frozen=True)
class ProjectPaths:
    project_file: Path
    target_trajectory: Path
    qpos_timeline: Path
    workspace: Path
    last_snapshot: Path
    session_log: Path


@dataclass(frozen=True)
class AutosavePaths:
    manifest: Path
    target_trajectory: Path
    qpos_timeline: Path
    workspace: Path


@dataclass
class GhostGUIProject:
    root_dir: Path
    metadata: dict
    source_schema_version: int = PROJECT_SCHEMA_VERSION

    def __post_init__(self):
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.metadata = migrate_project_metadata(self.metadata)
        self._validate_metadata_paths()

    @classmethod
    def create(cls, root_dir, project_name, model_key, model_name=None):
        root_dir = Path(root_dir).expanduser().resolve()
        now = utc_now_iso()
        metadata = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_name": project_name,
            "created_at": now,
            "modified_at": now,
            "application": {
                "name": "GhostGUI",
                "project_format": "ghostgui.project.v2",
            },
            "robot": {
                "model_key": model_key,
                "model_name": model_name or model_key,
            },
            "files": dict(DEFAULT_PROJECT_FILES),
            "autosave_files": dict(DEFAULT_AUTOSAVE_FILES),
            "persistence": {
                "transaction_format": "ghostgui.transaction.v1",
            },
        }
        project = cls(root_dir=root_dir, metadata=metadata)
        project.ensure_directories()
        project.save_metadata(update_modified=False)
        project.append_session_event(
            "project_created",
            {
                "project_name": project_name,
                "model_key": model_key,
                "model_name": model_name or model_key,
            },
        )
        return project

    @classmethod
    def open(cls, path):
        path = Path(path).expanduser()
        project_file = path / PROJECT_FILENAME if path.is_dir() else path
        project_file = project_file.resolve()
        recover_project_transactions(project_file.parent)
        metadata = _read_json(project_file)
        try:
            source_schema_version = int(metadata.get("schema_version", 0))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProjectFormatError(
                "project schema_version must be an integer"
            ) from exc
        metadata = migrate_project_metadata(metadata)
        return cls(
            root_dir=project_file.parent,
            metadata=metadata,
            source_schema_version=source_schema_version,
        )

    def _validate_metadata_paths(self):
        seen = set()
        for section_name, defaults in (
            ("files", DEFAULT_PROJECT_FILES),
            ("autosave_files", DEFAULT_AUTOSAVE_FILES),
        ):
            mapping = self.metadata.get(section_name)
            if mapping is None:
                mapping = {}
                self.metadata[section_name] = mapping
            if not isinstance(mapping, dict):
                raise ProjectFormatError(
                    f"project {section_name} must be a JSON object"
                )
            for key, default in defaults.items():
                mapping.setdefault(key, default)
            for key, relative in mapping.items():
                if not isinstance(relative, str) or not relative.strip():
                    raise ProjectFormatError(
                        f"project {section_name}.{key} must be a non-empty string"
                    )
                resolved = safe_project_path(
                    self.root_dir,
                    relative,
                    label=f"project {section_name}.{key}",
                )
                if resolved == self.project_file:
                    raise ProjectFormatError(
                        f"project {section_name}.{key} cannot overwrite metadata"
                    )
                if resolved in seen:
                    raise ProjectFormatError(
                        f"project file path is used more than once: {relative}"
                    )
                seen.add(resolved)

    @property
    def project_file(self):
        return self.root_dir / PROJECT_FILENAME

    @property
    def project_name(self):
        return self.metadata.get("project_name") or self.root_dir.stem

    @property
    def model_key(self):
        return self.metadata.get("robot", {}).get("model_key", "g1")

    @property
    def paths(self):
        return ProjectPaths(
            project_file=self.project_file,
            target_trajectory=self.resolve_file("target_trajectory"),
            qpos_timeline=self.resolve_file("qpos_timeline"),
            workspace=self.resolve_file("workspace"),
            last_snapshot=self.resolve_file("last_snapshot"),
            session_log=self.resolve_file("session_log"),
        )

    @property
    def autosave_paths(self):
        return AutosavePaths(
            manifest=self.resolve_autosave_file("manifest"),
            target_trajectory=self.resolve_autosave_file("target_trajectory"),
            qpos_timeline=self.resolve_autosave_file("qpos_timeline"),
            workspace=self.resolve_autosave_file("workspace"),
        )

    def resolve_file(self, key):
        relative = self.metadata.get("files", {}).get(key)
        if relative is None:
            relative = DEFAULT_PROJECT_FILES[key]
            self.metadata.setdefault("files", {})[key] = relative
        return safe_project_path(
            self.root_dir,
            relative,
            label=f"project files.{key}",
        )

    def resolve_autosave_file(self, key):
        relative = (
            self.metadata.get("autosave_files", {}).get(key)
            or DEFAULT_AUTOSAVE_FILES[key]
        )
        self.metadata.setdefault("autosave_files", {})[key] = relative
        return safe_project_path(
            self.root_dir,
            relative,
            label=f"project autosave_files.{key}",
        )

    def ensure_directories(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for path in self.paths.__dict__.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        for path in self.autosave_paths.__dict__.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "exports").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "logs").mkdir(parents=True, exist_ok=True)

    def update_robot(self, model_key, model_name=None):
        self.metadata.setdefault("robot", {})
        self.metadata["robot"]["model_key"] = model_key
        self.metadata["robot"]["model_name"] = model_name or model_key

    def save_metadata(self, update_modified=True):
        self.ensure_directories()
        self.metadata = migrate_project_metadata(self.metadata)
        self.metadata["schema_version"] = PROJECT_SCHEMA_VERSION
        if update_modified:
            self.metadata["modified_at"] = utc_now_iso()
        _write_json(self.project_file, self.metadata)

    def append_session_event(self, event, details=None):
        self.ensure_directories()
        record = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "event": str(event),
        }
        if details:
            record["details"] = _json_safe(details)
        with self.paths.session_log.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        return record

    def read_session_log(self, limit=None):
        records = []
        try:
            with self.paths.session_log.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except FileNotFoundError:
            return []
        if limit is None:
            return records
        return records[-int(limit):]

    def write_trajectory(self, trajectory):
        _write_json(self.paths.target_trajectory, trajectory.to_project_dict())

    def read_trajectory_dict(self, autosave=False):
        path = (
            self.autosave_paths.target_trajectory
            if autosave else self.paths.target_trajectory
        )
        return _read_json(path)

    def save_qpos_timeline(self, timeline):
        _atomic_timeline_save(timeline, self.paths.qpos_timeline)

    def load_qpos_timeline(self, timeline, autosave=False):
        path = (
            self.autosave_paths.qpos_timeline
            if autosave else self.paths.qpos_timeline
        )
        timeline.load_npz(path)

    def write_workspace(self, workspace_state):
        _write_json(self.paths.workspace, workspace_state)

    def read_workspace(self, autosave=False):
        path = self.autosave_paths.workspace if autosave else self.paths.workspace
        return _read_json(path)

    def write_autosave(
        self,
        trajectory,
        timeline,
        workspace_state,
        model_key,
        model_name=None,
    ):
        self.ensure_directories()
        transaction = ProjectSaveTransaction(self.root_dir)
        transaction.stage_json(
            self.autosave_paths.target_trajectory,
            trajectory.to_project_dict(),
        )
        transaction.stage_timeline(
            self.autosave_paths.qpos_timeline,
            timeline,
        )
        transaction.stage_json(
            self.autosave_paths.workspace,
            workspace_state,
        )
        transaction.stage_json(
            self.autosave_paths.manifest,
            {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "autosaved_at": utc_now_iso(),
                "robot": {
                    "model_key": model_key,
                    "model_name": model_name or model_key,
                },
            },
        )
        transaction.commit()

    def save_bundle(self, trajectory, timeline, workspace_state):
        """Atomically replace the coherent project document files."""
        self.ensure_directories()
        metadata = migrate_project_metadata(self.metadata)
        metadata["schema_version"] = PROJECT_SCHEMA_VERSION
        metadata["modified_at"] = utc_now_iso()

        transaction = ProjectSaveTransaction(self.root_dir)
        transaction.stage_json(
            self.paths.target_trajectory,
            trajectory.to_project_dict(),
        )
        if timeline is not None:
            transaction.stage_timeline(self.paths.qpos_timeline, timeline)
        transaction.stage_json(self.paths.workspace, workspace_state)
        transaction.stage_json(self.project_file, metadata)
        transaction.commit()
        self.metadata = metadata
        self.source_schema_version = PROJECT_SCHEMA_VERSION

    def read_autosave_manifest(self):
        return _read_json(self.autosave_paths.manifest)

    def autosave_exists(self):
        paths = self.autosave_paths
        return (
            paths.manifest.exists()
            and paths.target_trajectory.exists()
            and paths.qpos_timeline.exists()
            and paths.workspace.exists()
        )

    def autosave_model_key(self):
        if not self.autosave_paths.manifest.exists():
            return self.model_key
        manifest = self.read_autosave_manifest()
        return manifest.get("robot", {}).get("model_key", self.model_key)

    def is_autosave_newer(self):
        if not self.autosave_exists():
            return False
        try:
            autosave_mtime = max(
                path.stat().st_mtime
                for path in self.autosave_paths.__dict__.values()
            )
            saved_mtime = max(
                path.stat().st_mtime
                for path in (
                    self.project_file,
                    self.paths.target_trajectory,
                    self.paths.qpos_timeline,
                    self.paths.workspace,
                )
                if path.exists()
            )
        except (OSError, ValueError):
            return False
        return autosave_mtime > saved_mtime

    def clear_autosave(self):
        for path in self.autosave_paths.__dict__.values():
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
