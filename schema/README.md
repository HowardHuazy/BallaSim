<!-- SPDX-License-Identifier: CC0-1.0 -->
# Shared contract files

SPDX-License-Identifier: CC0-1.0. Every file in this folder and below it (the JSON Schema, the validator and the
golden exchanges, whose data files cannot carry a notice of their own) is dedicated to the public domain under CC0
1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/legalcode). CC0 grants no patent or trademark
rights (Section 4(a) of its legal code). The BallaSim engine repository and the BallaSim Bridge add-on repository
carry identical copies.

- `blender_exchange-1.json`: the JSON Schema of `exchange.json` in a `scene.bsx/` exchange,
  `ballasim.blender_exchange` 1.0 (BallaSim `docs/design/phase3/blender.md` §3.4). Version 1.0 keys only: a 1.0
  reader refuses unknown keys, so the schema does too.
- `validate.py`: the validator both repositories test with (the JSON Schema 2020-12 keywords the schema uses,
  standard library only; it refuses a schema that uses any other keyword).
  `python validate.py blender_exchange-1.json golden/blender-4.2/exchange.json`.
- `golden/blender-<major>.<minor>/`: an exchange written by the add-on in that Blender (`exchange.json`,
  `arrays.npz`, `textures/<sha256>.npy`): the d5 scene (a subsurf at its render levels, a bevel shown for render
  only), eye-hidden and viewport-disabled objects, a curve, a geometry-nodes instancer and a textured material; a
  camera, a point light in a light group, a world, and a keyed object in a collection with a semantic class.
  The add-on's tests reproduce it in the same Blender version; `BALLASIM_BLESS_GOLDEN=1` rewrites it.
