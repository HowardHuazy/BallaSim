# SPDX-License-Identifier: CC0-1.0
"""A small JSON Schema validator for the shared contract files (BallaSim and the BallaSim Bridge add-on), standard
library only: the 2020-12 keywords the schemas in this folder use. A schema that uses any other keyword is refused,
so a schema cannot silently ask for a check this validator does not make. One rule is stricter than 2020-12, as the
BallaSim reader is: an ``integer`` is a JSON number written without a fraction or exponent (1.0 is not one).

    python validate.py SCHEMA.json DOCUMENT.json...     # prints every error, exit 1 when there is one

or ``errors(document, schema)`` -> list of "path: message" strings (empty when the document is valid).
"""

from __future__ import annotations

import json
import re
import sys

_IGNORED = {"$schema", "$id", "$comment", "title", "description", "$defs", "default", "examples"}
_KEYWORDS = {"type", "enum", "const", "properties", "required", "additionalProperties", "items", "prefixItems",
             "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength",
             "pattern", "anyOf", "$ref"}
_TYPES = {"null": lambda v: v is None, "boolean": lambda v: isinstance(v, bool),
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "string": lambda v: isinstance(v, str), "array": lambda v: isinstance(v, list),
          "object": lambda v: isinstance(v, dict)}


class SchemaError(ValueError):
    """The schema itself uses something this validator does not check."""


_PATTERNS: dict[str, re.Pattern] = {}


def _pattern(p: str) -> re.Pattern:
    """A 2020-12 ``pattern`` (ECMA-262) as a Python regex: ``$`` outside a class is ``\\Z``, since ECMA's ``$`` matches
    only at the end of the string and Python's also before a final newline."""
    if p not in _PATTERNS:
        out, i, in_class = [], 0, False
        while i < len(p):
            c = p[i]
            if c == "\\":
                out.append(p[i:i + 2])
                i += 2
                continue
            if in_class and c == "]":
                in_class = False
            elif not in_class and c == "[":
                in_class = True
            out.append("\\Z" if c == "$" and not in_class else c)
            i += 1
        _PATTERNS[p] = re.compile("".join(out))
    return _PATTERNS[p]


def _equal(a, b) -> bool:
    """JSON equality: 1 == 1.0, but true is not 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local references are supported: {ref}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def errors(doc, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    root = schema if root is None else root
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: not allowed"]
    unknown = set(schema) - _KEYWORDS - _IGNORED
    if unknown:
        raise SchemaError(f"unsupported keywords at {path}: {sorted(unknown)}")
    out: list[str] = []
    if "$ref" in schema:
        out += errors(doc, _resolve(schema["$ref"], root), root, path)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](doc) for t in types):
            return out + [f"{path}: {json.dumps(doc)[:60]} is not of type {' or '.join(types)}"]
    if "const" in schema and not _equal(doc, schema["const"]):
        out.append(f"{path}: must be {json.dumps(schema['const'])}")
    if "enum" in schema and not any(_equal(doc, e) for e in schema["enum"]):
        out.append(f"{path}: {json.dumps(doc)[:60]} is not one of {json.dumps(schema['enum'])}")
    if "anyOf" in schema:
        branches = [errors(doc, s, root, path) for s in schema["anyOf"]]
        if all(branches):
            out.append(f"{path}: matches no alternative ({'; '.join(min(branches, key=len))})")
    if _TYPES["number"](doc):
        for key, bad in (("minimum", lambda x, b: x < b), ("maximum", lambda x, b: x > b),
                         ("exclusiveMinimum", lambda x, b: x <= b), ("exclusiveMaximum", lambda x, b: x >= b)):
            if key in schema and bad(doc, schema[key]):
                out.append(f"{path}: {doc} violates {key} {schema[key]}")
    if isinstance(doc, str):
        if "minLength" in schema and len(doc) < schema["minLength"]:
            out.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not _pattern(schema["pattern"]).search(doc):
            out.append(f"{path}: {doc[:60]!r} does not match {schema['pattern']}")
    if isinstance(doc, list):
        if "minItems" in schema and len(doc) < schema["minItems"]:
            out.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(doc) > schema["maxItems"]:
            out.append(f"{path}: more than {schema['maxItems']} items")
        prefix = schema.get("prefixItems", [])
        for i, item in enumerate(doc):
            sub = prefix[i] if i < len(prefix) else schema.get("items", True)
            out += errors(item, sub, root, f"{path}[{i}]")
    if isinstance(doc, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in doc:
                out.append(f"{path}: misses {key}")
        extra = schema.get("additionalProperties", True)
        for key, value in doc.items():
            if key in props:
                out += errors(value, props[key], root, f"{path}.{key}")
            elif extra is False:
                out.append(f"{path}: unknown key {key}")
            else:
                out += errors(value, extra, root, f"{path}.{key}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    with open(argv[0], encoding="utf-8") as f:
        schema = json.load(f)
    bad = 0
    for name in argv[1:]:
        with open(name, encoding="utf-8") as f:
            found = errors(json.load(f), schema)
        for e in found:
            print(f"{name}: {e}")
        bad += bool(found)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
