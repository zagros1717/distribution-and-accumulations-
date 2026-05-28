# ---- BTC Flow Intelligence — frontend (Next.js standalone) ----
# Railway-safe Dockerfile with explicit npm registry and longer fetch timeouts.

FROM node:20-slim AS deps
WORKDIR /app

ENV NPM_CONFIG_REGISTRY=https://registry.npmjs.org/
ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* ./

RUN npm config set registry https://registry.npmjs.org/ \
    && npm ci --fetch-timeout=120000 --fetch-retries=5

FROM node:20-slim AS builder
WORKDIR /app

ENV NPM_CONFIG_REGISTRY=https://registry.npmjs.org/
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["sh", "-c", "node server.js -p ${PORT:-3000}"]
