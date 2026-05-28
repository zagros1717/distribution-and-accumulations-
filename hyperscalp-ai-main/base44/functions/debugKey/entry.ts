import { ethers } from 'npm:ethers@6.11.1';

Deno.serve(async (req) => {
  try {
    const raw = Deno.env.get('ss');
    if (!raw) return Response.json({ error: 'ss not set' });

    let pk = raw.trim();
    const hadPrefix = pk.startsWith('0x');
    if (!hadPrefix) pk = '0x' + pk;
    const lengthBeforePad = pk.length;
    pk = '0x' + pk.slice(2).padStart(64, '0');

    const wallet = new ethers.Wallet(pk);

    return Response.json({
      rawLength: raw.length,
      hadPrefix,
      lengthBeforePad,
      lengthAfterPad: pk.length,
      derivedAddress: wallet.address,
      first6: raw.slice(0, 6),
      last4: raw.slice(-4),
    });
  } catch (e) {
    return Response.json({ error: e.message });
  }
});