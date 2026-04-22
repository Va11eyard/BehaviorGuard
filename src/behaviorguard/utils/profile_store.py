"""
Profile Store
=============
File-system backed persistence for UserProfile objects.

Profiles are stored as JSON files in a configurable directory.
Each profile is keyed by user_id and stored as:

    <store_dir>/<user_id>.json

The store is intentionally simple — for production, swap in a database
adapter (Postgres, DynamoDB, Redis) using the same interface.

Privacy note: profiles contain only statistical summaries (means, stds,
topic word lists), never raw message text.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from behaviorguard.models import UserProfile


_SAFE_USER_ID = re.compile(r"[^\w\-.]")  # only alphanumeric, dash, underscore, dot


def _sanitize(user_id: str) -> str:
    """Sanitize user_id to a safe filename component."""
    return _SAFE_USER_ID.sub("_", user_id)


class ProfileStore:
    """
    File-system backed store for UserProfile objects.

    Profiles are written as pretty-printed JSON files and can be read back
    atomically.  Atomic writes use a temp-file + rename pattern to avoid
    corrupt profiles on crash.

    Args:
        store_dir: Directory where profiles are stored.  Created if it does
                   not exist.
    """

    def __init__(self, store_dir: str | Path = "profiles"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save(self, profile: UserProfile) -> Path:
        """
        Persist a profile to disk.

        Writes atomically via a temp file to avoid partial writes.

        Args:
            profile: The UserProfile to persist.

        Returns:
            Path to the written file.
        """
        filename = self._path_for(profile.user_id)
        tmp_path = filename.with_suffix(".tmp")

        data = profile.model_dump()
        data["_saved_at"] = datetime.now(timezone.utc).isoformat()

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Atomic rename
        shutil.move(str(tmp_path), str(filename))
        return filename

    def load(self, user_id: str) -> Optional[UserProfile]:
        """
        Load a profile from disk.

        Args:
            user_id: The user identifier.

        Returns:
            UserProfile if found, else None.
        """
        path = self._path_for(user_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_saved_at", None)
        return UserProfile.model_validate(data)

    def delete(self, user_id: str) -> bool:
        """
        Delete a profile from disk.

        Args:
            user_id: The user identifier.

        Returns:
            True if the file was deleted, False if it did not exist.
        """
        path = self._path_for(user_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, user_id: str) -> bool:
        """Return True if a profile for user_id exists on disk."""
        return self._path_for(user_id).exists()

    # ── Listing ──────────────────────────────────────────────────────────────

    def list_user_ids(self) -> List[str]:
        """Return sorted list of all user_ids that have a stored profile."""
        return sorted(p.stem for p in self.store_dir.glob("*.json"))

    def __iter__(self) -> Iterator[UserProfile]:
        """Iterate over all stored profiles."""
        for user_id in self.list_user_ids():
            profile = self.load(user_id)
            if profile is not None:
                yield profile

    def __len__(self) -> int:
        return len(list(self.store_dir.glob("*.json")))

    # ── Convenience ──────────────────────────────────────────────────────────

    def load_or_cold_start(self, user_id: str, account_age_days: int = 0) -> UserProfile:
        """
        Load a profile from disk, or create a cold-start profile if none exists.

        Args:
            user_id:          User identifier.
            account_age_days: Used only when creating a cold-start profile.

        Returns:
            Existing UserProfile or a fresh cold-start profile.
        """
        from behaviorguard.profile_manager import ProfileManager

        profile = self.load(user_id)
        if profile is None:
            pm = ProfileManager()
            profile = pm.cold_start_profile(user_id, account_age_days)
        return profile

    def save_and_update(self, profile: UserProfile, message) -> UserProfile:
        """
        Incrementally update a profile with a new message and persist it.

        Convenience wrapper around ProfileManager.update_profile() + save().

        Args:
            profile: Current UserProfile.
            message: A MessageRecord instance.

        Returns:
            The updated (and persisted) UserProfile.
        """
        from behaviorguard.profile_manager import ProfileManager

        pm = ProfileManager()
        updated = pm.update_profile(profile, message)
        self.save(updated)
        return updated

    # ── Internal ─────────────────────────────────────────────────────────────

    def _path_for(self, user_id: str) -> Path:
        return self.store_dir / f"{_sanitize(user_id)}.json"

    def __repr__(self) -> str:
        return f"ProfileStore(store_dir={self.store_dir!r}, profiles={len(self)})"
