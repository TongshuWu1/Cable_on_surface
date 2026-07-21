"""Dataset metadata shared by PIDNet collection, review, and training."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid


DATASET_MANIFEST_VERSION = 2
DATASET_MANIFEST_NAME = "dataset_manifest.json"
DATASET_SPLITS = ("train", "val", "test")


def utc_now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_identifier(value, fallback="session"):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return text or str(fallback)


def new_session_id(prefix="session"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_identifier(prefix)}_{stamp}_{uuid.uuid4().hex[:6]}"


def unique_capture_stem(session_id):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"cable_{sanitize_identifier(session_id)}_{stamp}_{uuid.uuid4().hex[:8]}"


def manifest_path(dataset_root):
    return Path(dataset_root) / DATASET_MANIFEST_NAME


def empty_manifest():
    return {
        "version": DATASET_MANIFEST_VERSION,
        "updated_at": utc_now_text(),
        "sessions": {},
        "items": {},
    }


def load_dataset_manifest(dataset_root):
    path = manifest_path(dataset_root)
    if not path.exists():
        return empty_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != DATASET_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest version {payload.get('version')!r}; "
            f"expected {DATASET_MANIFEST_VERSION}: {path}"
        )
    payload.setdefault("sessions", {})
    payload.setdefault("items", {})
    return payload


def save_dataset_manifest(dataset_root, manifest):
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    manifest = dict(manifest)
    manifest["version"] = DATASET_MANIFEST_VERSION
    manifest["updated_at"] = utc_now_text()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def register_dataset_item(
    dataset_root,
    stem,
    session_id,
    split,
    verified=False,
    negative=False,
    source_path=None,
    notes="",
    annotation=None,
):
    split = str(split).strip().lower()
    if split not in DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset split: {split!r}")
    session_id = sanitize_identifier(session_id)
    stem = sanitize_identifier(stem, fallback="frame")
    manifest = load_dataset_manifest(dataset_root)
    sessions = manifest["sessions"]
    session = dict(sessions.get(session_id, {}))
    previous_split = session.get("split")
    if previous_split is not None and previous_split != split:
        raise ValueError(
            f"Session {session_id!r} is assigned to {previous_split!r}, not {split!r}. "
            "Move the whole session instead of splitting adjacent frames."
        )
    session.setdefault("created_at", utc_now_text())
    session["split"] = split
    sessions[session_id] = session

    old_item = dict(manifest["items"].get(stem, {}))
    item = {
        "session_id": session_id,
        "split": split,
        "verified": bool(verified),
        "negative": bool(negative),
        "created_at": old_item.get("created_at", utc_now_text()),
        "updated_at": utc_now_text(),
        "notes": str(notes or ""),
    }
    if source_path is not None:
        item["source_path"] = str(source_path)
    if annotation is not None:
        # Round-trip through JSON so the manifest never receives mutable GUI
        # objects or values that cannot be reproduced in the dataset snapshot.
        item["annotation"] = json.loads(json.dumps(annotation))
    elif "annotation" in old_item:
        item["annotation"] = old_item["annotation"]
    manifest["items"][stem] = item
    active_sessions = {entry.get("session_id") for entry in manifest["items"].values()}
    for stored_session_id in tuple(sessions):
        if stored_session_id not in active_sessions:
            sessions.pop(stored_session_id, None)
    save_dataset_manifest(dataset_root, manifest)
    return item


def remove_dataset_item_metadata(dataset_root, stem):
    manifest = load_dataset_manifest(dataset_root)
    removed = manifest["items"].pop(str(stem), None)
    if removed is not None:
        active_sessions = {item.get("session_id") for item in manifest["items"].values()}
        for session_id in tuple(manifest["sessions"]):
            if session_id not in active_sessions:
                manifest["sessions"].pop(session_id, None)
        save_dataset_manifest(dataset_root, manifest)
    return removed


def metadata_for_stem(dataset_root, stem):
    return dict(load_dataset_manifest(dataset_root).get("items", {}).get(str(stem), {}))


def move_session_split(dataset_root, session_id, new_split):
    """Move every saved item in one capture session to a single split."""
    root = Path(dataset_root)
    session_id = sanitize_identifier(session_id)
    new_split = str(new_split).strip().lower()
    if new_split not in DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset split: {new_split!r}")
    manifest = load_dataset_manifest(root)
    items = {
        stem: item
        for stem, item in manifest.get("items", {}).items()
        if sanitize_identifier(item.get("session_id")) == session_id
    }
    moves = []
    for stem, item in items.items():
        old_split = str(item.get("split", "")).lower()
        if old_split == new_split:
            continue
        if old_split not in DATASET_SPLITS:
            raise ValueError(f"Invalid stored split for {stem}: {old_split!r}")
        for folder, suffix in (("images", ".png"), ("masks", ".png"), ("masks_layers", ".npz")):
            source = root / folder / old_split / f"{stem}{suffix}"
            target = root / folder / new_split / f"{stem}{suffix}"
            if target.exists() and target.resolve() != source.resolve():
                raise FileExistsError(f"Refusing to overwrite session move target: {target}")
            if source.exists():
                moves.append((source, target))
    completed_moves = []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            completed_moves.append((source, target))
        for item in items.values():
            item["split"] = new_split
            item["updated_at"] = utc_now_text()
        session = dict(manifest.get("sessions", {}).get(session_id, {}))
        session["split"] = new_split
        session.setdefault("created_at", utc_now_text())
        manifest.setdefault("sessions", {})[session_id] = session
        save_dataset_manifest(root, manifest)
    except Exception:
        for source, target in reversed(completed_moves):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                target.replace(source)
        raise
    return len(items), len(moves)


def session_split_conflicts(manifest):
    observed = {}
    for stem, item in manifest.get("items", {}).items():
        session_id = sanitize_identifier(item.get("session_id"))
        split = str(item.get("split", "")).lower()
        observed.setdefault(session_id, {}).setdefault(split, []).append(stem)
    return {
        session_id: splits
        for session_id, splits in observed.items()
        if len({split for split in splits if split in DATASET_SPLITS}) > 1
    }


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataset_snapshot(dataset_root, pairs_by_split):
    root = Path(dataset_root).resolve()
    manifest = load_dataset_manifest(root)
    files = []
    digest = hashlib.sha256()
    selected_stems = set()
    for split in DATASET_SPLITS:
        for image_path, mask_path in pairs_by_split.get(split, ()):
            selected_stems.add(Path(image_path).stem)
            layer_path = root / "masks_layers" / split / f"{Path(mask_path).stem}.npz"
            for path in (Path(image_path), Path(mask_path), layer_path):
                resolved = path.resolve()
                relative = resolved.relative_to(root).as_posix()
                checksum = file_sha256(resolved)
                size = resolved.stat().st_size
                files.append({"path": relative, "bytes": size, "sha256": checksum})
                digest.update(relative.encode("utf-8"))
                digest.update(checksum.encode("ascii"))
    selected_items = {
        stem: manifest.get("items", {}).get(stem, {})
        for stem in sorted(selected_stems)
    }
    selected_session_ids = {
        sanitize_identifier(item.get("session_id"))
        for item in selected_items.values()
    }
    selected_metadata = {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "sessions": {
            session_id: manifest.get("sessions", {}).get(session_id, {})
            for session_id in sorted(selected_session_ids)
        },
        "items": selected_items,
    }
    metadata_bytes = json.dumps(selected_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    digest.update(metadata_sha256.encode("ascii"))
    return {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "dataset_sha256": digest.hexdigest(),
        "metadata_sha256": metadata_sha256,
        "files": files,
        "metadata": selected_metadata,
    }
