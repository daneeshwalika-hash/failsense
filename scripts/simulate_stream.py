#!/usr/bin/env python
"""Entry point of the stream simulator; the logic lives in failsense.simulation.stream.

uv run python scripts/simulate_stream.py --interval 1 --max-readings 200 --drift-start 60
"""

from failsense.simulation.stream import main

if __name__ == "__main__":
    main()
