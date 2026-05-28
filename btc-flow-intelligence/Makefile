.PHONY: up down backend-test frontend-build audit zip

up:
	docker compose up --build

down:
	docker compose down

backend-test:
	cd apps/backend && python -m pytest -q

frontend-build:
	cd apps/frontend && npm ci && npm run build

audit:
	cd apps/frontend && npm audit --omit=dev || true

zip:
	rm -rf apps/frontend/node_modules apps/frontend/.next
	cd .. && zip -qr btc-flow-intelligence-production.zip btc-flow-intelligence -x 'btc-flow-intelligence/.git/*'
