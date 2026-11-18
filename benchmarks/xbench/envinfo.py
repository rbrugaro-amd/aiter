import json, os, subprocess, sys
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
def info():
    import torch, triton
    d = dict(torch=torch.__version__, hip=torch.version.hip, torch_git=torch.version.git_version,
             triton=triton.__version__, rocm=sh("cat /opt/rocm/.info/version"),
             hipblaslt=sh("ls /opt/rocm/lib/libhipblaslt.so.* 2>/dev/null | head -3; dpkg -l 2>/dev/null | grep -i hipblaslt | awk '{print $2,$3}'"),
             aiter_head=sh("git -C /sgl-workspace/aiter rev-parse HEAD"),
             sglang_head=sh("git -C /sgl-workspace/sglang rev-parse HEAD"),
             hip_visible=os.environ.get("HIP_VISIBLE_DEVICES"),
             aiter_env={k: v for k, v in os.environ.items() if k.startswith(("AITER", "SGLANG_USE_AITER"))},
             triton_cache=os.environ.get("TRITON_CACHE_DIR"),
             device=torch.cuda.get_device_name(0), cus=torch.cuda.get_device_properties(0).multi_processor_count)
    try:
        import aiter
        d["aiter_version"] = getattr(aiter, "__version__", sh("pip show amd-aiter | grep Version"))
        d["aiter_file"] = aiter.__file__
    except Exception as e:
        d["aiter_import"] = repr(e)
    try:
        import sglang; d["sglang_version"] = sglang.__version__
    except Exception as e:
        d["sglang_import"] = repr(e)
    return d
if __name__ == "__main__":
    print(json.dumps(info(), indent=1))
