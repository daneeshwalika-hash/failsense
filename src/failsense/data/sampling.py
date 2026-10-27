"""Draw random cut-off points per vehicle from the labelled training readouts.

Why this module exists
----------------------
The SCANIA validation and test splits give **one** label per truck: the class
at the *last* readout of a history that was cut off at a random point. The
training split, in contrast, contains the *full* history of each truck, and
``failsense.data.labels.build_labels`` gives a label to *every* readout.

If we trained on every readout, the training examples would not look like
the questions the model is asked later (train/serving skew):

* trucks with long histories would dominate the training set,
* readouts of one truck are almost copies of each other,
* the age distribution and the share of positives would differ from
  validation/test.

So we imitate the validation set: for every truck we pick ``k`` random
readouts and treat each one as "the moment we ask the model". Features for
such a cut-off are later built **only from readouts up to that cut-off**
(``features.py``, 30.10), which keeps the method free of future information.

Rules implemented here
----------------------
1. Only readouts with a **known** label are candidates. Unknown labels
   (``<NA>``, censored trucks close to the end of the study) are never drawn.
2. ``k`` (``n_per_vehicle``) comes from the config. ``None`` means "keep all
   known readouts" (option A in the session notes), ``1`` imitates the
   validation set exactly (option B), larger values are a compromise
   (option C). The best value is chosen with cross validation, not guessed.
3. A truck with fewer known readouts than ``k`` contributes all of them;
   a readout is never drawn twice (sampling without replacement).
4. The draw is reproducible: the same ``seed`` always gives the same rows.
5. The result keeps the original index and row order of the input, so it can
   be joined back to the readouts and features without surprises.

Leakage note: several cut-offs of the same truck are allowed because cross
validation uses ``StratifiedGroupKFold`` with ``groups=vehicle_id``. All
cut-offs of one truck therefore always land in the same fold.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

# Reuse the column names from labels.py instead of repeating the strings.
# One source of truth: if a name changes, it changes in one place only.
from failsense.data.labels import CLASS_COL, VEHICLE_COL

logger = logging.getLogger(__name__)


def validate_n_per_vehicle(n_per_vehicle: int | None) -> int | None:
    """Check that ``n_per_vehicle`` is ``None`` or a positive integer.

    Raises:
        TypeError: If the value is not an int (``bool`` is rejected too,
            because ``True`` is technically an int in Python and would
            silently mean ``1``).
        ValueError: If the value is zero or negative.
    """
    if n_per_vehicle is None:
        return None
    if isinstance(n_per_vehicle, bool) or not isinstance(n_per_vehicle, int):
        raise TypeError(f"n_per_vehicle must be an int or None, got {n_per_vehicle!r}")
    if n_per_vehicle < 1:
        raise ValueError(f"n_per_vehicle must be >= 1, got {n_per_vehicle}")
    return n_per_vehicle


def sampling_params_from_config(config: Mapping[str, Any]) -> tuple[int | None, int]:
    """Read ``sampling.n_cutoffs_per_vehicle`` and ``sampling.seed`` from the config.

    In YAML, ``null`` becomes ``None`` in Python and means "keep all".

    Returns:
        ``(n_per_vehicle, seed)``.
    """
    try:
        section = config["sampling"]
        n_per_vehicle = section["n_cutoffs_per_vehicle"]
        seed = section["seed"]
    except KeyError as exc:
        raise KeyError("config needs 'sampling.n_cutoffs_per_vehicle' and 'sampling.seed'") from exc
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"sampling.seed must be an int, got {seed!r}")
    return validate_n_per_vehicle(n_per_vehicle), seed


def sample_cutoffs(
    labels: pd.DataFrame,
    n_per_vehicle: int | None,
    seed: int,
) -> pd.DataFrame:
    """Pick up to ``n_per_vehicle`` random labelled readouts per vehicle.

    Args:
        labels: Output of ``build_labels`` (needs ``vehicle_id`` and
            ``proximity_class``; other columns are kept unchanged).
        n_per_vehicle: Cut-offs per vehicle, or ``None`` for all known rows.
        seed: Seed for the random generator (reproducible draws).

    Returns:
        A subset of the rows of ``labels`` with the same columns, the
        original index and the original row order.

    Raises:
        ValueError: If required columns are missing or ``n_per_vehicle < 1``.
        TypeError: If ``n_per_vehicle`` is not an int or ``None``.
    """
    k = validate_n_per_vehicle(n_per_vehicle)
    missing = [c for c in (VEHICLE_COL, CLASS_COL) if c not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required columns: {missing}")

    # Step 1: drop unknown labels. `notna()` is False for <NA>, so censored
    # readouts close to the end of the study can never be drawn.
    known = labels[labels[CLASS_COL].notna()]
    n_unknown = len(labels) - len(known)

    if k is None:
        logger.info(
            "Sampling disabled: keeping all %d known readouts (%d unknown dropped)",
            len(known),
            n_unknown,
        )
        return known

    # Step 2: shuffle the row *positions* (0, 1, 2, ...) with a seeded
    # generator. np.random.default_rng is NumPy's modern, recommended API;
    # the same seed always produces the same permutation.
    rng = np.random.default_rng(seed)
    shuffled_pos = rng.permutation(len(known))
    shuffled = known.iloc[shuffled_pos]

    # Step 3: inside each vehicle, number the shuffled rows 0, 1, 2, ...
    # (like ROW_NUMBER() OVER (PARTITION BY vehicle_id ORDER BY random) in
    # SQL Server) and keep the first k. Because the order is random, this is
    # a random draw of k rows per vehicle without replacement. A vehicle with
    # fewer than k rows simply keeps all of them, so no special case is needed.
    keep = shuffled.groupby(VEHICLE_COL, sort=False).cumcount().to_numpy() < k

    # Step 4: sort the selected positions to restore the original row order.
    selected_pos = np.sort(shuffled_pos[keep])
    sampled = known.iloc[selected_pos]

    logger.info(
        "Sampled %d cut-offs from %d vehicles (k=%d, seed=%d); %d unknown readouts dropped",
        len(sampled),
        sampled[VEHICLE_COL].nunique(),
        k,
        seed,
        n_unknown,
    )
    return sampled
