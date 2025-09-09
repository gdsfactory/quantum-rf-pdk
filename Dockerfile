FROM ghcr.io/astral-sh/uv:python3.13-slim

# Create user for binder
ARG NB_USER=notebook-user
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# Install deps. with uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra dev --extra docs && \
    uv pip install jupyterlab

# Install rest of project separately allowing optimal layer caching
COPY . ${HOME}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev --extra docs && \
    uv pip install jupyterlab

ENV PATH="/${HOME}/.venv/bin:$PATH"

# Reset the entrypoint
ENTRYPOINT []