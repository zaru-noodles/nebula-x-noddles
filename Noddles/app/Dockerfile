# syntax=docker/dockerfile:1

FROM node:22-trixie-slim AS dependencies
WORKDIR /workspace/app/train-condition-monitoring
COPY app/train-condition-monitoring/package.json app/train-condition-monitoring/package-lock.json ./
RUN npm ci

FROM node:22-trixie-slim AS builder
WORKDIR /workspace/app/train-condition-monitoring
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=dependencies /workspace/app/train-condition-monitoring/node_modules ./node_modules
COPY app/train-condition-monitoring/ ./
RUN npm run build

FROM node:22-trixie-slim AS runner
WORKDIR /srv

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=8080 \
    MODEL_ROOT=/srv/Optional_Items \
    PYTHON_EXECUTABLE=/opt/venv/bin/python \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY --from=builder --chown=node:node /workspace/app/train-condition-monitoring/.next/standalone ./
COPY --from=builder --chown=node:node /workspace/app/train-condition-monitoring/.next/static ./.next/static
COPY --from=builder --chown=node:node /workspace/app/train-condition-monitoring/public ./public
COPY --chown=node:node Optional_Items/ ./Optional_Items/

USER node
EXPOSE 8080

CMD ["node", "server.js"]

