"""Thin scene/run adapters for baked radiance (docs/design/phase3/scans.md §§3.4, 3.7).

The renderer owns radiance and shadows; these helpers carry source metadata into the resolver and run record.
"""


def facts(bundle) -> dict:
    """The resolver's baked facts, independent of run options."""
    materials = [b.materials[j] for b in bundle.bodies for j in b.mesh.used_materials()]
    return {"has_baked": any(m.shading == "baked" for m in materials),
            "has_proxies": any(light.proxy for light in bundle.lights),
            "all_lights_proxy": all(light.proxy for light in bundle.lights),
            "lit_emissive": any(m.shading == "lit" and bool((m.emissive != 0).any()) for m in materials)}


def record(bundle, scale: float) -> dict | None:
    """The rendered bundle's ``baked`` run block; absent for a scene without baked materials."""
    if not facts(bundle)["has_baked"]:
        return None
    groups = [g.name for g in bundle.light_groups]
    materials = []
    for body in bundle.bodies:
        for j in sorted(body.mesh.used_materials()):
            material = body.materials[j]
            if material.shading != "baked":
                continue
            shares = [1.0, *([0.0] * (len(groups) - 1))] if material.baked_groups is None else material.baked_groups
            materials.append({"body": body.id, "body_name": body.name, "material": int(j),
                              "K": float(material.baked_scale), "groups": dict(zip(groups, map(float, shares))),
                              "source": "f16" if material.texture_linear is not None else
                                        "srgb8" if material.texture is not None else "constant"})
    proxies = [{"light": j, "kind": light.kind, "group": groups[light.group]}
               for j, light in enumerate(bundle.lights) if light.proxy]
    fill = {g: sum(float(light.intensity) for light in bundle.lights
                   if light.proxy and light.kind == "ambient" and groups[light.group] == g) for g in groups}
    return {"materials": materials, "baked_scale": float(scale), "scan_bodies": [b.id for b in bundle.bodies if b.scan],
            "proxies": proxies, "fill": fill, "proxy_blocked": {}}
