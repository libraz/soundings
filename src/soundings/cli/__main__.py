"""So that `python -m soundings.cli` runs the same thing the installed script does."""

from __future__ import annotations

import sys

from . import main

sys.exit(main())
