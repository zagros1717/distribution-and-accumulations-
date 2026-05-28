/**
 * Fetches real account balance and open positions from Hyperliquid.
 * Wallet address is read from BotConfig — no hardcoded address (FIX #11).
 */
import { createClientFromRequest } from 'npm:@base44/sdk@0.8.25';

Deno.serve(async (req) => {
  try {
    const base44 = createClientFromRequest(req);
    const configs = await base44.entities.BotConfig.list();
    const address = configs?.[0]?.wallet_address?.trim();

    if (!address) {
      return Response.json({ error: 'BotConfig.wallet_address is empty — set it in Settings' }, { status: 400 });
    }

    const res = await fetch('https://api.hyperliquid.xyz/info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'clearinghouseState', user: address }),
    });
    const data = await res.json();

    const accountValue = parseFloat(data?.marginSummary?.accountValue || 0);
    const positions = (data?.assetPositions || [])
      .filter(p => parseFloat(p.position?.szi || 0) !== 0)
      .map(p => ({
        coin: p.position.coin,
        size: parseFloat(p.position.szi),
        entryPrice: parseFloat(p.position.entryPx),
        unrealizedPnl: parseFloat(p.position.unrealizedPnl),
        leverage: parseFloat(p.position.leverage?.value || 1),
      }));

    return Response.json({ address, accountValue, positions });
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
});