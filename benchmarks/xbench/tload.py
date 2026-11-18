"""Load a template host entry from /work (wt_next) with its manifest sha256 checked."""
import hashlib, json, pathlib, re, sys, types, importlib.util
from functools import lru_cache

WORK = pathlib.Path("/work")
PACKS = WORK / "src/kernel_packs/gfx950"
PROFILES = WORK / "src/kernel_packs/profiles"


def manifest_sha(profile, source_file):
    """sha256 recorded for `source_file` in a profile json (rows or source_dependencies)."""
    d = json.loads((PROFILES / profile).read_text())
    shas = {r["source_sha256"] for r in d.get("specializations", []) if r.get("source_file") == source_file}
    if not shas:
        raise KeyError(f"{source_file} not in {profile}")
    assert len(shas) == 1, shas
    return shas.pop()


@lru_cache(None)
def load(pack_rel, entry, profile=None, source_file=None):
    """pack_rel relative to src/kernel_packs/gfx950 (e.g. 'glm52_fp8/qkv_a_norm_r3/qkv_a_norm_template.py').
    profile: profile json relative to profiles/ whose rows name this file (sha check)."""
    if "kernel_packs" not in sys.modules:
        pkg = WORK / "src/kernel_packs/__init__.py"
        spec = importlib.util.spec_from_file_location("kernel_packs", pkg)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["kernel_packs"] = mod
        spec.loader.exec_module(mod)
    path = PACKS / pack_rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if profile is not None:
        want = manifest_sha(profile, source_file or "/".join(pack_rel.split("/")[1:]))
        if want != digest:
            raise ValueError(f"{path} sha {digest} != manifest {want}")
    name = f"xb_tpl_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    pkg = "_kt_pkg_" + re.sub(r"\W", "_", str(path.parent.resolve()))
    if pkg not in sys.modules:
        parent = types.ModuleType(pkg)
        parent.__path__ = [str(path.parent)]
        sys.modules[pkg] = parent
    module.__package__ = pkg
    sys.modules[name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, entry)
    fn._xb_sha256 = digest
    return fn
