# =========================
# PennyLane Docker Image for Setonix GPU cluster
# docker buildx build --platform linux/amd64 -f pennylane-setonixgpu.dockerfile --target wheel-out --build-arg PY_VER=3.11 --build-arg LIGHTNING_VERSION=master --build-arg PENNYLANE_VERSION=master --build-arg AMD_ARCH=AMD_GFX90A --build-arg GCC_HOST_VER=11 --output type=local,dest=./wheelhouse . 
# or download from quay.io/pawsey/pennylane-gracehopper:0.0.1
# singularity shell -e --rocm pennylane-setonixgpu.sif
# Build-time arguments (override at build as needed)
# - PY_VER: choose 3.11 or 3.12 (both supported on Ubuntu 22.04 via deadsnakes)
# - AMD_ARCH: Kokkos architecture flag (e.g., AMD_GFX90A for MI200, AMD_GFX942 for MI300)
# - GCC_HOST_VER: host GCC major version (for hipcc --gcc-install-dir path on x86_64 jammy)
# =========================
ARG PY_VER=3.11
ARG PENNYLANE_VERSION=master
ARG LIGHTNING_VERSION=master
ARG AMD_ARCH=AMD_GFX90A
ARG GCC_HOST_VER=11

# =========================
# 1) Build stage (ROCm dev image with compilers and headers)
#    - x86_64 host
#    - HIP/ROCm 6.2.4 toolchain present
#    - Installs Python ${PY_VER}, venv, build toolchain
# =========================
FROM docker.io/rocm/dev-ubuntu-22.04:6.4.1-complete AS build-base

ARG PY_VER
ARG PENNYLANE_VERSION
ARG LIGHTNING_VERSION
ARG AMD_ARCH
ARG GCC_HOST_VER
ARG DEBIAN_FRONTEND=noninteractive

# Use explicit python binary name everywhere to be version-agnostic
ENV PYBIN=python${PY_VER}
ENV PIP=pip

# OS + Python toolchain
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
    software-properties-common gnupg ca-certificates curl git rsync \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
 && apt-get update \
 && apt-get install --no-install-recommends -y \
    ${PYBIN} ${PYBIN}-venv ${PYBIN}-dev python3-pip \
    build-essential cmake ninja-build libgomp1 \
    gcc-${GCC_HOST_VER} g++-${GCC_HOST_VER} \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set GCC-11 as default compiler
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_HOST_VER} 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_HOST_VER} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_HOST_VER}

# Build venv (isolates Python toolchain and wheels)
ENV VIRTUAL_ENV=/opt/venv-build
RUN ${PYBIN} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Python build helpers
RUN ${PIP} install --no-cache-dir --upgrade pip \
 && ${PIP} install --no-cache-dir build cmake ninja wheel setuptools>=75.8.1 toml

# Fetch pennylane-lightning sources at requested branch/tag
RUN git clone --depth 1 --branch ${LIGHTNING_VERSION} https://github.com/PennyLaneAI/pennylane-lightning.git /opt/pennylane-lightning
WORKDIR /opt/pennylane-lightning

# This script updates pyproject for backend toggles (tolerate absence on old revs)
RUN ${PYBIN} scripts/configure_pyproject_toml.py || true

# =========================
# 2) Build lightning-qubit wheel
#    - Pure CPU backend; useful for internal/tests and parity with upstream builds
# =========================
FROM build-base AS build-wheel-lightning-qubit
WORKDIR /opt/pennylane-lightning

# Ensure a clean rebuild without prior wheel in env
RUN ${PIP} uninstall -y pennylane-lightning || true

# Build wheel (qubit backend is default when PL_BACKEND is unset)
RUN ${PYBIN} -m build --wheel

# =========================
# 3) Build lightning-kokkos (HIP) wheel
#    - Kokkos HIP enablement; OpenMP optional (ON here as many users rely on it)
#    - Important: set --gcc-install-dir for hipcc on x86_64 jammy
# =========================
FROM build-base AS build-wheel-lightning-kokkos-hip
WORKDIR /opt/pennylane-lightning
ARG AMD_ARCH
ARG GCC_HOST_VER

# Kokkos/HIP env
ENV CMAKE_PREFIX_PATH=/opt/rocm:${CMAKE_PREFIX_PATH}
ENV CXX=hipcc
ENV PL_BACKEND=lightning_kokkos

# Build Kokkos-HIP wheel with selected AMD arch and OpenMP enabled
# Following official Dockerfile format exactly (using GCC-11)
RUN ${PIP} uninstall -y pennylane-lightning
RUN ${PYBIN} scripts/configure_pyproject_toml.py || true
RUN CMAKE_ARGS="-DKokkos_ENABLE_SERIAL:BOOL=ON \
    -DKokkos_ENABLE_OPENMP:BOOL=ON \
    -DKokkos_ENABLE_HIP:BOOL=ON \
    -DKokkos_ARCH_${AMD_ARCH}=ON \
    -DCMAKE_CXX_FLAGS='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/${GCC_HOST_VER}/'" \
    ${PYBIN} -m build --wheel

# =========================
# 4) Runtime stage - Install lightning-kokkos HIP backend
#    Following official Dockerfile format
# =========================
FROM quay.io/pawsey/rocm-mpich-base:rocm6.4.1-mpich3.4.3-ubuntu24.04 AS runtime

ARG PY_VER
ARG PENNYLANE_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYBIN=python${PY_VER}

# Minimal runtime dependencies + Python
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    software-properties-common \
    git \
    libgomp1 \
    python3-pip \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
    ${PYBIN} \
    ${PYBIN}-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN ${PYBIN} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"


# Copy wheels from build stages
COPY --from=build-wheel-lightning-kokkos-hip /opt/pennylane-lightning/dist/ /opt/wheels/
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /opt/wheels/

# Install wheels and PennyLane
RUN pip install --force-reinstall --no-cache-dir /opt/wheels/pennylane_lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}
