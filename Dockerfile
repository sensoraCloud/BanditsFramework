FROM plds/data-science-utils:1.3.0

ARG GITHUB_TOKEN
ARG VAULT_PASSWORD

ENV DEBIAN_FRONTEND noninteractive
ENV PATH="/opt/program:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONUNBUFFERED=TRUE
ENV TERM linux

WORKDIR /opt/program

# requirements
ARG GITHUB_TOKEN
COPY requirements.txt .
RUN pip install -U pip \
    && pip install -r requirements.txt

# vault
ARG VAULT_PASSWORD
COPY .env.enc .
RUN echo $VAULT_PASSWORD > .vaultpwd \
    && ansible-vault decrypt --vault-password-file=.vaultpwd --output=.env .env.enc \
    && rm .vaultpwd

COPY . .
ARG MODEL_VERSION
ENV MODEL_VERSION $MODEL_VERSION