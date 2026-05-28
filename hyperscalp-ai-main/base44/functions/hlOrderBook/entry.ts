/**
 * Fetches real order book data from Hyperliquid.
 * POST body: { coin: string }
 */
Deno.serve(async (req) => {
  try {
    const { coin } = await req.json();

    const res = await fetch('https://api.hyperliquid.xyz/info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'l2Book', coin }),
    });
    const data = await res.json();

    const levels = data?.levels || [[], []];
    const bidVolume = levels[0].slice(0, 10).reduce((s, l) => s + parseFloat(l.sz || 0), 0);
    const askVolume = levels[1].slice(0, 10).reduce((s, l) => s + parseFloat(l.sz || 0), 0);

    return Response.json({ coin, bidVolume, askVolume });
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
});