#!/usr/bin/env python
"""Entry point of the single frozen test evaluation; the logic is failsense.evaluate_test.

NOT TO BE RUN without Daneesh's approval, and only after `make freeze` on the final commit
(see docs/test_evaluation_plan.md):

    uv run python scripts/evaluate_test_once.py --approved-single-run

Rehearsal first (`make rehearse`): the same code path on the VALIDATION split, without a lock and
without opening a test file; it writes nothing inside the repository:

    uv run python scripts/evaluate_test_once.py --rehearse-on-validation
"""

from failsense.evaluate_test import main

if __name__ == "__main__":
    main()
