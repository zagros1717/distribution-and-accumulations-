/**
 * Fetches real-time prices from Hyperliquid for selected coins.
 * POST body: { coins: string[] }
 */
import { createClientFromRequest } from 'npm:@base44/sdk@0.8.25';

Deno.serve(async (req) => {
  try {
    const body = await req.json().catch(() => ({}));
    const requestedCoins = body.coins || null;

    const [midsRes, metaRes] = await Promise.all([
      fetch('https://api.hyperliquid.xyz/info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'allMids' }),
      }),
      fetch('https://api.hyperliquid.xyz/info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'meta' }),
      }),
    ]);
    const mids = await midsRes.json();
    const meta = await metaRes.json();

    // Only valid perp pairs (filter out delisted + spot indices like @123)
    const perpCoins = (meta?.universe || [])
      .filter(c => !c.isDelisted)
      .map(c => c.name)
      .sort();

    const prices = {};
    const coinsToUse = requestedCoins || perpCoins;
    for (const coin of coinsToUse) {
      const raw = mids[coin];
      if (raw) prices[coin] = parseFloat(raw);
    }

    return Response.json({ prices, allCoins: perpCoins });
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
});