"""
Build specimen-disjoint train/test splits for the kidney-stone datasets
(data version "v2", second revision of the MedIA manuscript).

Why: the train/test partitions released with the private datasets are
patch-disjoint but NOT specimen-disjoint. Different photographs of the same
stone (surface 1..6, section 1..5), colour-corrected copies (``...corr``,
``...bis``) and augmented copies (``m1 - <id> - ...``) end up on both sides.
For the MyStone release, the original loader also read the binary
segmentation masks (``labels/``) and the full frames (``images/``) as class
samples; v2 keeps the 256x256 patches only, as described in the paper.

What it does:
  * JONATHAN / DAUDON: pool the released train+test patches of every view
    (MIX, SEC, SUR), assign each source image to a specimen group, and draw a
    class-stratified, group-disjoint ~80/20 split. The assignment is made once
    per dataset on specimen groups, so a specimen is on the same side in every
    view.
  * MYSTONE: keep the released split, patches only; drop from the test side
    the (rare) specimens that also occur in train.

Specimen keys are recovered from file names with heuristics (see
``specimen_key``). They are conservative (they may merge distinct specimens,
never split one), but they still need confirmation by the data owners:
``data/splits/specimen_groups_review.csv`` lists every group for that review.

Run from the repository root (deterministic, safe to re-run)::

    python scripts/build_specimen_splits.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "splits")
SPLIT_SEED = 0          # fixed: the split is part of the data, not of the run
TEST_FRACTION = 0.20

DATASETS = {
    "JONATHAN": "Jonathan El-Beze (w256 1k v1)",
    "DAUDON": "Michel Daudon (w256 1k v1)",
}
VIEWS = ("MIX", "SEC", "SUR")
MYSTONE_DIR = os.path.join("MyStone", "MyStone (w256, 1k)")
IMG_EXT = (".png", ".jpg", ".jpeg")


def source_image(fname):
    """Strip the patch index: '25962 - surface 2-17.png' -> '25962 - surface 2'."""
    return re.sub(r"-\d+(\s*\(\d+\))?\.(png|jpe?g)$", "", fname, flags=re.I)


def specimen_key(dataset, src):
    """Specimen group of a source image, plus the rule that produced it."""
    if dataset == "JONATHAN":
        # '25962 - surface 2', 'm1 - 25962 - surface 2' (augmented copy)
        m = re.search(r"(?<!\d)(\d{4,6})(?!\d)", src)
        if m:
            return m.group(1), "id"
    else:
        # Specimen numbers: 72222, N4065, T71883, LC3584 (prefix dropped so that
        # 'N19226' and '19226' meet); camera frame numbers DSC_1191 as fallback.
        m = re.search(r"(?<![A-Za-z0-9])(?:N|T|LC)?(\d{4,6})(?!\d)", src.replace("DSC_", "DSC"))
        if m and not src[max(0, m.start() - 3):m.start()].upper().endswith("DSC"):
            return m.group(1), "id"
        d = re.search(r"DSC_?(\d+)", src)
        if d:
            return "DSC" + d.group(1), "dsc"
    # Fallback: the name itself without correction/duplicate suffixes, cut at the
    # first '-' (merges 'TypVa_6' with 'TypVa_6-30', 'TypIVd_4' with 'TypIVd_4bis').
    k = src.lower()
    k = re.sub(r"(corr\d*|bis|det\d*|\(\d+\))", "", k)
    k = k.split("-")[0].strip(" _")
    return "name:" + k, "name"


def list_patches(dataset_dir, view):
    """(relpath, class_name, released_split) for every patch of one view."""
    rows = []
    for split in ("train", "test"):
        base = os.path.join(DATA, dataset_dir, view, split)
        for cls in sorted(os.listdir(base)):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.startswith(".") or not f.lower().endswith(IMG_EXT):
                    continue
                rows.append((os.path.relpath(os.path.join(cdir, f), DATA), cls, split))
    return rows


def stratified_group_split(group_class_counts, rng):
    """Greedy, class-stratified, group-disjoint split.

    Each group is attached to its majority class; within every class the groups
    are visited in random order and a group goes to test only if it keeps the
    class within 10% above TEST_FRACTION of its patches (large specimens would
    otherwise overshoot the target).
    """
    by_class = defaultdict(list)
    for g, cnt in group_class_counts.items():
        by_class[cnt.most_common(1)[0][0]].append(g)
    class_total = Counter()
    for cnt in group_class_counts.values():
        class_total.update(cnt)
    test_groups = set()
    for cls in sorted(by_class):
        groups = sorted(by_class[cls])
        rng.shuffle(groups)
        target = TEST_FRACTION * class_total[cls]
        taken = 0
        for g in groups:
            size = group_class_counts[g][cls]
            if taken == 0 or taken + size <= 1.1 * target:
                test_groups.add(g)
                taken += size
            if taken >= target:
                break
    return test_groups


def build_private(name, dirname, review_rows, summary):
    rng = np.random.RandomState(SPLIT_SEED)
    per_view = {v: list_patches(dirname, v) for v in VIEWS}
    # Specimen groups are shared by all views; counts use the MIX view (the one
    # every experiment trains on) so that the 80/20 target holds there.
    group_of = {}
    counts = defaultdict(Counter)
    info = defaultdict(lambda: {"rule": "", "sources": set(), "released": set()})
    for view, rows in per_view.items():
        for rel, cls, rsplit in rows:
            src = source_image(os.path.basename(rel))
            g, rule = specimen_key(name, src)
            group_of[rel] = g
            subtype = cls.split("Subtype_")[-1]
            if view == "MIX":
                counts[g][subtype] += 1
            info[g]["rule"] = rule
            info[g]["sources"].add(src)
            info[g]["released"].add(rsplit)
    test_groups = stratified_group_split(counts, rng)

    for view, rows in per_view.items():
        out = os.path.join(OUT, f"{name}_{view}_v2.csv")
        n = Counter()
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "class_name", "split", "group", "released_split"])
            for rel, cls, rsplit in rows:
                g = group_of[rel]
                split = "test" if g in test_groups else "train"
                w.writerow([rel, cls, split, g, rsplit])
                n[(split, cls)] += 1
        summary[f"{name}_{view}"] = {f"{s}/{c}": v for (s, c), v in sorted(n.items())}

    for g, d in sorted(info.items()):
        review_rows.append([name, g, d["rule"], "test" if g in test_groups else "train",
                            "+".join(sorted(d["released"])),
                            ";".join(sorted(counts[g])), len(d["sources"]),
                            " | ".join(sorted(d["sources"])[:4])])
    leaked = sum(1 for d in info.values() if len(d["released"]) > 1)
    summary[f"{name}_groups"] = {"groups": len(info), "test_groups": len(test_groups),
                                 "groups_on_both_sides_in_release": leaked,
                                 "rules": dict(Counter(d["rule"] for d in info.values()))}


def build_mystone(review_rows, summary):
    rows = []
    for split in ("train", "test"):
        base = os.path.join(DATA, MYSTONE_DIR, split)
        for cls in sorted(os.listdir(base)):
            pdir = os.path.join(base, cls, "patch")
            if not os.path.isdir(pdir):
                continue
            for f in sorted(os.listdir(pdir)):
                if f.startswith(".") or not f.lower().endswith(IMG_EXT):
                    continue
                m = re.match(r"rs_([A-Za-z]+\d+)", f)
                g = m.group(1) if m else "name:" + source_image(f)
                rows.append((os.path.relpath(os.path.join(pdir, f), DATA), cls, split, g))
    train_groups = {g for _, _, s, g in rows if s == "train"}
    n = Counter()
    dropped = 0
    with open(os.path.join(OUT, "MYSTONE_v2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "class_name", "split", "group", "released_split"])
        for rel, cls, split, g in rows:
            if split == "test" and g in train_groups:
                dropped += 1
                continue
            w.writerow([rel, cls, split, g, split])
            n[(split, cls)] += 1
    summary["MYSTONE"] = {f"{s}/{c}": v for (s, c), v in sorted(n.items())}
    summary["MYSTONE"]["dropped_test_patches_shared_specimen"] = dropped
    for g in sorted({r[3] for r in rows}):
        sides = {r[2] for r in rows if r[3] == g}
        review_rows.append(["MYSTONE", g, "id", "+".join(sorted(sides)), "+".join(sorted(sides)),
                            ";".join(sorted({r[1] for r in rows if r[3] == g})), "", ""])


def main():
    os.makedirs(OUT, exist_ok=True)
    review, summary = [], {}
    for name, dirname in DATASETS.items():
        build_private(name, dirname, review, summary)
    build_mystone(review, summary)
    with open(os.path.join(OUT, "specimen_groups_review.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "group", "rule", "v2_split", "released_split",
                    "subtypes", "n_source_images", "example_sources"])
        w.writerows(review)
    with open(os.path.join(OUT, "summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
