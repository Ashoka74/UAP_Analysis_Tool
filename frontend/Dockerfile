# React frontend image: Vite build → nginx.
#
#   docker build -t ufosint-frontend ./frontend
#   docker run -p 8080:80 ufosint-frontend
#
# nginx serves the SPA and proxies /api/* to the backend service (container
# name `api` on the compose network), mirroring the Vite dev proxy — so
# VITE_API_BASE stays unset and the app calls a same-origin /api. For a split
# deploy against a remote backend instead, build with:
#   docker build --build-arg VITE_API_BASE=https://my-api.example.com ./frontend

# ── Build stage ─────────────────────────────────────────────────────────────
# NOTE (WSL2): if `npm ci` dies with "Exit handler never called!" and exits 0
# (leaving a broken node_modules that only fails later at `tsc: not found`),
# the container's bridge-network DNS is at fault — build with:
#   docker build --network host -t ufosint-frontend ./frontend
FROM node:20-slim AS build

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Inlined by Vite at build time (see src/api/client.ts). Empty = same-origin /api.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

# ── Serve stage ─────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost/ || exit 1
