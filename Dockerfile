# Note, to download from ghcr.io you may need to authenticate with docker login, e.g.
#     echo $(gh auth token) | docker login ghcr.io -u "$(gh api user | jq -r .login)" --password-stdin

FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim@sha256:2ab87c099cd28eeacc2d1122e8e2e020651fdf92c68d277caa003c3bba0627cc

# Create user for binder
ARG NB_USER=notebook-user
ARG NB_UID=1001
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER}
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_TOOL_DIR=/usr/local/lib/uv-tools \
    UV_CACHE_DIR=${HOME}/.cache/uv

# Install pinned just from the prebuilt rust-just wheel (kept for interactive use)
RUN uv tool install rust-just==1.58.0

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    mkdir -p ${HOME} && \
    chown -R ${USER}:${USER} ${HOME} && \
    chown -R ${USER}:${USER} /usr/local/bin /usr/local/lib

# Apt dependencies for gdsfactory & KLayout
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git=1:2.47.3-0+deb13u1 libexpat1=2.7.1-2 libexpat1-dev=2.7.1-2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
USER ${USER}

# First install only dependencies with cache mount
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --all-extras --group docs

# Copy source code, install project and convert jupytext scripts to notebooks
COPY --chown=${USER}:${USER} . ${HOME}
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    uv sync --locked --all-extras --group docs && \
    uv run --no-sync jupytext --to ipynb qpdk/samples/*.py

# Set PATH to include virtual environment
ENV PATH="${HOME}/.venv/bin:$PATH"

# Expose Jupyter Lab port
EXPOSE 8888

SHELL ["/bin/bash", "-c"]
# Jupyter starts with authentication disabled only when the container is managed by
# JupyterHub (mybinder.org injects JUPYTERHUB_API_TOKEN into the singleuser container):
# the hub proxies and authenticates users itself. A direct `docker run` has no such
# token and starts with Jupyter's default generated token, printed to the container
# logs. The container runs as a non-root user by default; --allow-root keeps
# `docker run --user 0` (used for bind-mount ownership workarounds) working.
# jupyter lab resolves from the pinned venv on PATH.
CMD ["/bin/bash", "-c", "if [ -n \"$JUPYTERHUB_API_TOKEN\" ]; then exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''; else exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root; fi"]
