FROM lmsysorg/sglang-rocm:v0.5.23-rocm724-mi35x-20261117
ARG TRITON_VERSION=3.8.0+gitdf3f91dd
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN /opt/venv/bin/python -m pip install --no-deps --force-reinstall --no-cache-dir \
      --index-url https://download.pytorch.org/whl/nightly/rocm7.2 "triton==${TRITON_VERSION}"
