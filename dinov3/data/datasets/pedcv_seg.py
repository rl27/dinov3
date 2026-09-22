# Downstream segmentation datasets for the pedestrian-POV encoder project.
#
# Two datasets, one shared 6-class taxonomy:
#
#     0 background   1 road   2 curb   3 sidewalk   4 crosswalk   5 terrain   255 void
#
# Both follow the ADE20K pattern in this directory: get_image_data / get_target return
# raw bytes and the pipeline's DenseTargetDecoder turns the target into a PIL mask, so
# these plug straight into dinov3/eval/segmentation via a "Vistas:split=VAL" descriptor.
#
# TAXONOMY WARNING. Class 0 is *background*, a real class -- not ADE20K's
# ignore-everything-unlabeled 0. Any config using these datasets must set
# `eval.reduce_zero_label: False`, and void is 255 rather than a class index.
# Note that dinov3/eval/segmentation/eval.py originally hardcoded reduce_zero_label=True
# in the metric regardless of config; that is patched locally (see the LOCAL PATCH note
# there). If you re-vendor upstream, re-apply it or every mIoU here is silently wrong.
#
# GROUP METADATA. Neither `get_target` nor the collate carries group information --
# upstream's segmentation transforms expect target to be a bare mask. Group-wise
# evaluation (worst-region mIoU on Vistas, per-session aggregation on SANPO) instead
# reads the parallel arrays `get_regions()` / `get_sessions()`, which are indexed the
# same way as the samples.

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset

# Shared taxonomy. Keep in sync with AGENTS.md.
CLASS_NAMES = ("background", "road", "curb", "sidewalk", "crosswalk", "terrain")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# Region label used for a Vistas image whose location we never recovered. It is NOT a
# region -- it is an absence of evidence, and the filtering logic below treats it as such.
UNKNOWN_REGION = "__unknown__"


class _VistasSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def dirname(self) -> str:
        return {
            _VistasSplit.TRAIN: "training",
            _VistasSplit.VAL: "validation",
            _VistasSplit.TEST: "testing",
        }[self]


class _SanpoSplit(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        return {_SanpoSplit.TRAIN: "train", _SanpoSplit.VAL: "val"}[self]


def _parse_region_list(value: Union[str, Sequence[str], None]) -> Optional[Tuple[str, ...]]:
    """"North_America,Africa" -> ("North America", "Africa").

    Underscores become spaces because the dataset descriptor is colon/comma delimited
    and region names contain spaces ("South America", "Northern Europe").
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [v for v in value.split(",") if v.strip()]
    else:
        parts = list(value)
    return tuple(p.strip().replace("_", " ") for p in parts) or None


@lru_cache(maxsize=2)
def _load_vistas_geo(geo_parquet: str) -> dict:
    """key (image stem) -> {continent, subregion, country}. Empty if the file is absent."""
    if not os.path.exists(geo_parquet):
        return {}
    import pandas as pd

    df = pd.read_parquet(geo_parquet, columns=["key", "continent", "subregion", "country"])
    return {
        r.key: {"continent": r.continent, "subregion": r.subregion, "country": r.country}
        for r in df.itertuples()
    }


class Vistas(ExtendedVisionDataset):
    """Mapillary Vistas with labels remapped to the 6-class taxonomy.

    Args:
        split: TRAIN (18,000) / VAL (2,000) / TEST (5,000, IMAGES ONLY -- the release
            ships no labels for testing, so TEST is export/inspection only and raises
            if you ask it for targets).
        root: project root; must contain ``vistas/`` and (for regions) ``data/``.
        region_field: which recovered geography to group by -- "continent" (6 values
            present) or "subregion" (19). Worst-group metrics get noisy fast as the
            group count rises; continent is the safer headline.
        region / exclude_region: comma-separated region names, ``_`` for spaces.
            Used to build hold-one-out-region protocols.
        keep_unknown_region: see the note on exclude_region below. Default False.
        geo_parquet: override the path to the recovered-geography table.
    """

    Split = Union[_VistasSplit]
    Labels = Union[Image.Image]

    def __init__(
        self,
        split: "Vistas.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
        region_field: str = "continent",
        region: Union[str, Sequence[str], None] = None,
        exclude_region: Union[str, Sequence[str], None] = None,
        keep_unknown_region: bool = False,
        geo_parquet: Optional[str] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        if region_field not in ("continent", "subregion", "country"):
            raise ValueError(f"region_field must be continent|subregion|country, got {region_field!r}")
        self.split = split
        self.region_field = region_field
        self._has_labels = split is not _VistasSplit.TEST

        image_dir = os.path.join(root, "vistas", split.dirname, "images")
        label_dir = os.path.join(root, "vistas", split.dirname, "converted_labels")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"{image_dir} not found")
        if self._has_labels and not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"{label_dir} not found -- labels must already be remapped to the 6-class taxonomy"
            )

        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg"))
        if self._has_labels:
            # An image with no mask would otherwise surface as a FileNotFoundError
            # thousands of iterations into a probe run.
            have = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".png")}
            missing = [s for s in stems if s not in have]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {split.dirname} images have no converted label "
                    f"(e.g. {missing[:3]}) -- refusing to build a partially-labelled split"
                )

        geo = _load_vistas_geo(geo_parquet or os.path.join(root, "data/vistas_geo/vistas_geo.parquet"))
        regions = [
            (geo.get(s) or {}).get(region_field) or UNKNOWN_REGION
            for s in stems
        ]

        keep_set = _parse_region_list(region)
        drop_set = _parse_region_list(exclude_region)
        if keep_set or drop_set:
            # A hold-one-out-region claim is about what the model never saw. An image
            # whose location we failed to recover is not evidence that it is outside the
            # held-out region -- only 69% of train+val geography is recovered, so keeping
            # unknowns in a "no Africa" split leaves unlabelled African images in it and
            # quietly weakens the very claim the split exists to support. Dropping them
            # is the honest default; keep_unknown_region=True is the explicit opt-out.
            selected = []
            for i, r in enumerate(regions):
                if r == UNKNOWN_REGION:
                    if not keep_unknown_region:
                        continue
                elif keep_set and r not in keep_set:
                    continue
                elif drop_set and r in drop_set:
                    continue
                selected.append(i)
            stems = [stems[i] for i in selected]
            regions = [regions[i] for i in selected]
            if not stems:
                raise ValueError(
                    f"region filter (region={region!r}, exclude_region={exclude_region!r}) "
                    f"left 0 images -- check the names against region_field={region_field!r}"
                )

        self._stems: Tuple[str, ...] = tuple(stems)
        self._regions: Tuple[str, ...] = tuple(regions)
        self._image_dir = image_dir
        self._label_dir = label_dir

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self._image_dir, self._stems[index] + ".jpg"), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        if not self._has_labels:
            raise RuntimeError(
                "Vistas TEST ships images only -- it has no segmentation labels and cannot be evaluated"
            )
        with open(os.path.join(self._label_dir, self._stems[index] + ".png"), "rb") as f:
            return f.read()

    def get_regions(self) -> Tuple[str, ...]:
        """Per-sample region label, index-aligned with the dataset. Drives worst-region mIoU."""
        return self._regions

    def get_keys(self) -> Tuple[str, ...]:
        """Per-sample Mapillary v3 image key (the filename stem)."""
        return self._stems

    def __len__(self) -> int:
        return len(self._stems)


class Sanpo(ExtendedVisionDataset):
    """SANPO segmentation with labels in the same 6-class taxonomy.

    Frames are video-correlated: consecutive frames of one session are near-duplicates,
    so a frame-level mean is effectively weighted by session length and its error bars
    are meaningless. Aggregate per session -- ``get_sessions()`` gives the grouping.

    Depth lives alongside as ``depth/<stem>.float16.gz`` at a different resolution to
    the images, and is handled by the separate depth reader rather than here.
    """

    Split = Union[_SanpoSplit]
    Labels = Union[Image.Image]

    def __init__(
        self,
        split: "Sanpo.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
        sessions: Union[str, Sequence[str], None] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        self.split = split
        base = os.path.join(root, "sanpo", split.dirname)
        self._image_dir = os.path.join(base, "images")
        self._label_dir = os.path.join(base, "segmentation")
        self._depth_dir = os.path.join(base, "depth")
        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(f"{self._image_dir} not found")

        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(self._image_dir) if f.endswith(".png"))
        have = {os.path.splitext(f)[0] for f in os.listdir(self._label_dir) if f.endswith(".png")}
        missing = [s for s in stems if s not in have]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} {split.dirname} frames have no segmentation label (e.g. {missing[:3]})"
            )

        if sessions is not None:
            wanted = set(sessions.split(",") if isinstance(sessions, str) else sessions)
            stems = [s for s in stems if _sanpo_session(s) in wanted]
            if not stems:
                raise ValueError(f"session filter {sorted(wanted)[:5]}... left 0 frames")

        self._stems: Tuple[str, ...] = tuple(stems)
        self._sessions: Tuple[str, ...] = tuple(_sanpo_session(s) for s in stems)

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self._image_dir, self._stems[index] + ".png"), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        with open(os.path.join(self._label_dir, self._stems[index] + ".png"), "rb") as f:
            return f.read()

    def get_sessions(self) -> Tuple[str, ...]:
        """Per-sample session id, index-aligned. Frames within a session are correlated."""
        return self._sessions

    def get_depth_path(self, index: int) -> str:
        return os.path.join(self._depth_dir, self._stems[index] + ".float16.gz")

    def get_keys(self) -> Tuple[str, ...]:
        return self._stems

    def __len__(self) -> int:
        return len(self._stems)


def _sanpo_session(stem: str) -> str:
    """"0xC_000000" -> "0xC". Frame index is everything after the last underscore."""
    return stem.rsplit("_", 1)[0]


def unique_groups(groups: Sequence[str]) -> List[str]:
    """Stable-ordered unique group labels, for building a group -> row index map."""
    seen, out = set(), []
    for g in groups:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out
