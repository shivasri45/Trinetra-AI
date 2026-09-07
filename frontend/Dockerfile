# Builds the dashboard and serves it from Caddy, which also fronts the API.
# Build context is the repository root, so paths are frontend/-prefixed.

FROM node:20-slim AS build
WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Vite inlines these at build time. VITE_TRINETRA_API defaults to the /api
# prefix that Caddy strips, keeping the dashboard same-origin with the API.
#
# WARNING: VITE_TRINETRA_API_KEY ends up in the built JavaScript in clear text.
# Anyone who can load the page can read it. Treat it as a deployment identifier,
# not a secret, and put real access control in front of the site if it is
# reachable from the internet.
ARG VITE_TRINETRA_API=/api
ARG VITE_TRINETRA_API_KEY
ENV VITE_TRINETRA_API=$VITE_TRINETRA_API \
    VITE_TRINETRA_API_KEY=$VITE_TRINETRA_API_KEY
RUN npm run build

FROM caddy:2.8-alpine AS runtime
COPY frontend/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 80 443
