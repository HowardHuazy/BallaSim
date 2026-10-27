"""Canonical scan-asset licence names (docs/design/phase3/scans.md §3.7).

Unknown licences keep their original text; callers warn on baked assets. Restricted flags describe source terms
and propagate to run records and exports, without changing the source licence or granting redistribution rights.
"""

import re

CANONICAL = ("CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-ND-4.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0",
             "CC-BY-NC-ND-4.0", "user-owned", "Replica-Research", "Matterport-Academic-EULA", "ScanNet-ToU",
             "ScanNet++-ToU")
RESTRICTED = {"Replica-Research", "Matterport-Academic-EULA", "ScanNet-ToU", "ScanNet++-ToU"}


def _key(text: str) -> str:
    return re.sub(r"[\s_]+", "-", text.strip().lower())


ALIASES = {_key(name): name for name in CANONICAL}
ALIASES.update({"cc0": "CC0-1.0", "own": "user-owned", "replica": "Replica-Research",
                "replica-dataset": "Replica-Research", "matterport": "Matterport-Academic-EULA",
                "matterport3d": "Matterport-Academic-EULA", "mp3d": "Matterport-Academic-EULA",
                "hm3d": "Matterport-Academic-EULA", "scannet": "ScanNet-ToU", "scannet++": "ScanNet++-ToU",
                "scannetpp": "ScanNet++-ToU"})


def canonical(license: str) -> str | None:
    """The table's canonical id, or None when the licence is unknown."""
    return ALIASES.get(_key(license)) if isinstance(license, str) else None


def records(bundle) -> list[dict]:
    """The source assets' terms from the bundle, the sole source of run.json ``licenses`` (§3.7)."""
    out = []
    assets = bundle.meta.get("assets", {})
    if isinstance(assets, dict):
        entries = assets.items()
    elif isinstance(assets, list):
        # Blender's hand-off uses AssetUse records, whose kind/index disambiguate names shared by different IDs.
        entries = ((f"{a.get('kind', 'asset')}:{a.get('index', i)}:{a.get('name', '')}", a)
                   for i, a in enumerate(assets) if isinstance(a, dict))
    else:
        return out
    for name, asset in entries:
        if not isinstance(asset, dict):
            continue
        licence = canonical(asset.get("license"))
        out.append({"asset": name, "license": licence or asset.get("license"),
                    "attribution": asset.get("attribution") or "; ".join(str(asset[k]) for k in ("author", "copyright")
                                                                        if asset.get(k)),
                    "scale_basis": asset.get("scale_basis"),
                    "restricted": bool(asset.get("restricted") or licence in RESTRICTED)})
    return out
