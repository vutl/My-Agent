"""Keep import-time app initialization away from the user's runtime data."""
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pytest


def pytest_configure(config):
    # Several endpoint tests import app.main during collection, before fixtures
    # can override settings. Worktrees may symlink data/ to the live library.
    from app.core.config import get_settings

    source = get_settings().sqlite_db_path
    directory = TemporaryDirectory(prefix="aya-backend-tests-")
    # Legacy catalog integration tests use the installed library. Preserve its
    # evidence with a read-only snapshot rather than migrating the live DB.
    if source.is_file():
        target = Path(directory.name) / "sqlite/app.db"
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        # Include uncheckpointed transactions without asking SQLite to create
        # a shared-memory file next to the read-only production database.
        journal = source.with_name(source.name + "-wal")
        if journal.is_file():
            shutil.copyfile(journal, target.with_name(target.name + "-wal"))
    patch = pytest.MonkeyPatch()
    patch.setenv("APP_DATA_DIR", directory.name)
    get_settings.cache_clear()

    def cleanup():
        patch.undo()
        get_settings.cache_clear()
        directory.cleanup()

    config.add_cleanup(cleanup)
