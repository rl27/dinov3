# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Unlabeled Mapillary street-scene crawl with per-image metadata.

The pretraining corpus for metadata-guided adaptation of a pedestrian-POV encoder. Returns
``(image, (label, _Metadata))`` so the collate fn can route metadata to guide heads. There are
no class labels -- this corpus is unlabeled by construction -- so ``label`` is a constant 0
placeholder, present only to satisfy the ``(label, metadata)`` contract that
``GuidedSSLMetaArch`` expects. Nothing consumes it.

Reads two artifacts produced by ``scripts/build_manifest.py``:

  manifest.parquet      one row per usable image, with guide columns precomputed
  guide_classes.json    string -> index maps, the single source of truth for class indices

Class indices come from ``guide_classes.json`` rather than being derived here. A prototypical
guide keeps one EMA centroid per class, so if an index shifted between runs -- because the
manifest was rebuilt and some rare camera make vanished -- every prototype would silently
re-bind to the wrong class. Deriving them from whatever rows happen to be present would
reintroduce exactly that hazard.

Corpus composition (``min_quality`` / ``vehicle_fraction``) is applied here rather than by
building a second manifest, so every arm reads the one manifest that carries the Vistas
contamination filter. Both are deterministic functions of the manifest's row order and
``sample_seed``, which matters because ``cache_dataset=true`` puts a ShardedInfiniteSampler
over this dataset: every rank builds its own copy and they must agree index-for-index.

Two things the manifest guarantees, both load-bearing (see AGENTS.md):
  * Vistas evaluation images and their sequence-mates are already removed, so pretraining on
    this corpus cannot contaminate the held-out-region robustness protocol.
  * ``year`` is clipped to 2011-2026, because unset camera clocks produce EXIF years as far
    back as 1970 which would otherwise become singleton noise classes for the adversarial
    year guide.
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


@dataclass
class _Metadata:
    """Per-image metadata. Field names MUST match ``guide.guides[*].name`` in the config."""

    # --- the two-guide reference recipe (mirrors vitl16_fmow_guided.yaml) ---
    sub_region: int          # UN subregion from lat/lon (19; Melanesia merged into ANZ) -- informative guide
    year: int                # capture year index, 2011-2026 (16 classes) -- adversarial guide
    # --- sweep candidates, one at a time (what FINO's Fig. 4 does) ---
    transport_mode: int      # walk / vehicle (2 classes)
    country: int             # crawl ISO country code (166 classes)
    camera_make: int         # camera manufacturer (526 classes)
    camera_type: int         # perspective / fisheye / ... (5 classes)
    month: int               # 0-indexed (12 classes)
    season: int              # hemisphere-corrected (4 classes)
    hour_local: int          # local solar hour from longitude (24 classes)
    coordinates: Tuple[float, float]  # (lat_norm, lon_norm) in [0, 1], for regression
    quality_score: float     # Mapillary's own score, roughly [0, 1]


class _Split(Enum):
    TRAIN = "train"          # the whole corpus
    WALK = "walk"            # pedestrian POV only
    VEHICLE = "vehicle"      # vehicle POV only


def _apply_vehicle_fraction(df, fraction: float, seed: int):
    """Subsample so that ``fraction`` of the returned rows are vehicle-POV.

    Keeps as many rows as the target fraction allows without duplicating any: whichever
    transport mode is not the binding constraint is kept whole and the other is subsampled.
    At ``fraction=0.0`` this is exactly ``split=WALK`` (SPEC.md arm C1); at ``1.0`` it is
    ``split=VEHICLE``. Row order is preserved, so the result is a subsequence of the
    manifest, not a shuffle.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"vehicle_fraction must be in [0, 1], got {fraction}")

    is_vehicle = (df["transport_mode"] == "vehicle").to_numpy()
    veh_pos = np.flatnonzero(is_vehicle)
    walk_pos = np.flatnonzero(~is_vehicle)
    n_v, n_w = len(veh_pos), len(walk_pos)

    # Keep all walk and take the vehicle rows the ratio asks for; if there are not enough
    # vehicle rows, keep them all and subsample walk instead.
    if fraction < 1.0 and round(n_w * fraction / (1.0 - fraction)) <= n_v:
        keep_w, keep_v = n_w, int(round(n_w * fraction / (1.0 - fraction)))
    elif fraction > 0.0:
        keep_w, keep_v = int(round(n_v * (1.0 - fraction) / fraction)), n_v
    else:
        keep_w, keep_v = n_w, 0

    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(walk_pos, size=keep_w, replace=False) if keep_w < n_w else walk_pos,
        rng.choice(veh_pos, size=keep_v, replace=False) if keep_v < n_v else veh_pos,
    ])
    chosen.sort()
    return df.iloc[chosen]


@lru_cache(maxsize=4)
def _load_manifest(
    root: str,
    split: _Split,
    min_quality: float = 0.0,
    vehicle_fraction: float = -1.0,
    sample_seed: int = 0,
):
    """Load the manifest once per (root, split, filters); cached because workers re-import."""
    import pandas as pd

    manifest = os.path.join(root, "manifest.parquet")
    classes_path = os.path.join(root, "guide_classes.json")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"{manifest} not found -- run scripts/build_manifest.py first. Without it the "
            "Vistas contamination filter has not been applied to this corpus."
        )
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"{classes_path} not found -- run scripts/build_manifest.py first.")

    with open(classes_path) as f:
        n_outputs = json.load(f)["_n_outputs"]

    df = pd.read_parquet(manifest)
    if split == _Split.WALK:
        df = df[df["transport_mode"] == "walk"]
    elif split == _Split.VEHICLE:
        df = df[df["transport_mode"] == "vehicle"]

    # Quality floor first, so the vehicle fraction is exact over the corpus actually used.
    # A missing score cannot be shown to clear the floor, so it is dropped rather than
    # defaulted to 0.0 the way the metadata field below is.
    if min_quality > 0.0:
        df = df[df["quality_score"].notna() & (df["quality_score"] >= min_quality)]

    if vehicle_fraction >= 0.0:
        if split != _Split.TRAIN:
            raise ValueError(
                f"vehicle_fraction is meaningless with split={split.value} -- that split has "
                "already fixed the transport mode."
            )
        df = _apply_vehicle_fraction(df, vehicle_fraction, sample_seed)

    df = df.reset_index(drop=True)

    paths = tuple(df["path"].tolist())

    def col(name, default=0):
        return df[name].fillna(default).astype("int64").to_numpy()

    meta = tuple(
        _Metadata(
            sub_region=int(sr), year=int(yr), transport_mode=int(tm), country=int(co),
            camera_make=int(mk), camera_type=int(ct), month=int(mo), season=int(se),
            hour_local=int(hl), coordinates=(float(la), float(lo)), quality_score=float(qs),
        )
        for sr, yr, tm, co, mk, ct, mo, se, hl, la, lo, qs in zip(
            col("subregion_id"), col("year_id"), col("transport_mode_id"), col("country_id"),
            col("camera_make_id"), col("camera_type_id"), col("month"), col("season"),
            col("hour_local"),
            df["lat_norm"].fillna(0.5).to_numpy(), df["lon_norm"].fillna(0.5).to_numpy(),
            df["quality_score"].fillna(0.0).to_numpy(),
        )
    )
    return paths, meta, n_outputs


class Mapillary(ExtendedVisionDataset):
    """Unlabeled Mapillary crawl returning ``(image, (label, _Metadata))``.

    Args:
        split: TRAIN (all), WALK, or VEHICLE.
        root: directory holding ``manifest.parquet`` and ``guide_classes.json``
            (i.e. the project's ``data/``).
        image_root: directory the manifest's relative ``path`` column is resolved against
            (the project root). Defaults to ``root/..``.
        with_metadata: TRUE (default) -> ``(image, (label, _Metadata))``. FALSE ->
            ``(image, label)``, needed by evaluators whose collate cannot batch the dataclass.
        min_quality: drop rows whose ``quality_score`` is below this floor (0.0 = keep all,
            the default). Rows with no score are dropped whenever a floor is set.
        vehicle_fraction: resample so this fraction of the corpus is vehicle-POV, applied
            after ``min_quality``. Negative (the default) leaves the natural mix untouched;
            it is only valid on ``split=TRAIN``.
        sample_seed: seed for the ``vehicle_fraction`` subsample. Must not vary across ranks.
    """

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Mapillary.Split" = _Split.TRAIN,
        root: str,
        image_root: str = "",
        with_metadata: bool = True,
        min_quality: float = 0.0,
        vehicle_fraction: float = -1.0,
        sample_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._split = split
        self._with_metadata = with_metadata
        super().__init__(root=root, **kwargs)
        self._image_root = image_root or os.path.dirname(os.path.abspath(root))
        self._image_paths, self._metadata, self.guide_n_outputs = _load_manifest(
            root, split, float(min_quality), float(vehicle_fraction), int(sample_seed)
        )

    @property
    def split(self) -> "Mapillary.Split":
        return self._split

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self._image_root, self._image_paths[index]), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        # label is a constant placeholder: this corpus has no class labels.
        if self._with_metadata:
            return 0, self._metadata[index]
        return 0

    def get_targets(self) -> np.ndarray:
        return np.zeros(len(self._image_paths), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._image_paths)
