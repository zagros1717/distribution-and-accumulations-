/**
 * Executes a trade on Hyperliquid (open or close a perpetual position).
 * POST body: { action: 'open'|'close', coin, direction, size_usd, price }
 *
 * Signing scheme: msgpack(action) + nonce(u64 BE) + vault byte(s) → keccak256 → EIP-712 Agent
 */
import { ethers } from 'npm:ethers@6.11.1';
import { encode as msgpackEncode } from 'npm:@msgpack/msgpack@3.0.0';

function addressToBytes(address) {
  const hex = address.slice(2).toLowerCase();
  const bytes = new Uint8Array(20);
  for (let i = 0; i < 20; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

function encodeUint64BE(value) {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setBigUint64(0, BigInt(value), false);
  return new Uint8Array(buf);
}

function floatToWire(x) {
  const rounded = Math.round(x * 1e8) / 1e8;
  let s = rounded.toPrecision(8);
  if (s.includes('e') || s.includes('E')) s = rounded.toString();
  if (s.includes('.')) s = s.replace(/\.?0+$/, '');
  return s;
}

function actionHash(action, vaultAddress, nonce) {
  const actionBytes = msgpackEncode(action);
  const nonceBytes = encodeUint64BE(nonce);
  const vaultByte = vaultAddress
    ? new Uint8Array([1, ...addressToBytes(vaultAddress)])
    : new Uint8Array([0]);

  const combined = new Uint8Array(actionBytes.length + nonceBytes.length + vaultByte.length);
  combined.set(actionBytes, 0);
  combined.set(nonceBytes, actionBytes.length);
  combined.set(vaultByte, actionBytes.length + nonceBytes.length);

  return ethers.keccak256(combined);
}

async function signL1Action(wallet, action, vaultAddress, nonce, isMainnet = true) {
  const hash = actionHash(action, vaultAddress, nonce);

  const phantomAgent = {
    source: isMainnet ? 'a' : 'b',
    connectionId: hash,
  };

  const domain = {
    chainId: 1337,
    name: 'Exchange',
    verifyingContract: '0x0000000000000000000000000000000000000000',
    version: '1',
  };
  const types = {
    Agent: [
      { name: 'source', type: 'string' },
      { name: 'connectionId', type: 'bytes32' },
    ],
  };

  const signature = await wallet.signTypedData(domain, types, phantomAgent);
  const sig = ethers.Signature.from(signature);
  return { r: sig.r, s: sig.s, v: sig.v };
}

Deno.serve(async (req) => {
  try {
    let privateKey = Deno.env.get('ss');
    if (!privateKey) return Response.json({ error: 'ss secret not set' }, { status: 500 });
    privateKey = privateKey.trim();
    if (!privateKey.startsWith('0x')) privateKey = '0x' + privateKey;

    const wallet = new ethers.Wallet(privateKey);
    const { action, coin, direction, size_usd, price } = await req.json();

    const metaRes = await fetch('https://api.hyperliquid.xyz/info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'meta' }),
    });
    const meta = await metaRes.json();

    if (!meta?.universe) {
      return Response.json({ error: 'Invalid metadata response' }, { status: 500 });
    }

    const coinIndex = meta.universe.findIndex(c => c.name === coin);
    if (coinIndex === -1) return Response.json({ error: `Unknown coin: ${coin}` }, { status: 400 });

    const szDecimals = meta.universe[coinIndex].szDecimals;
    const sizeInCoin = parseFloat((size_usd / price).toFixed(szDecimals));

    const isBuy = action === 'open' ? direction === 'long' : direction === 'short';
    // Aggressive limit price for IoC fill (slippage tolerance ~0.2%)
    const rawLimit = isBuy ? price * 1.002 : price * 0.998;
    const limitPrice = parseFloat(rawLimit.toPrecision(5));

    const orderAction = {
      type: 'order',
      orders: [{
        a: coinIndex,
        b: isBuy,
        p: floatToWire(limitPrice),
        s: floatToWire(sizeInCoin),
        r: action === 'close',
        t: { limit: { tif: 'Ioc' } },
      }],
      grouping: 'na',
    };

    const nonce = Date.now();
    const signature = await signL1Action(wallet, orderAction, null, nonce, true);

    const payload = {
      action: orderAction,
      nonce,
      signature,
      vaultAddress: null,
    };

    const res = await fetch('https://api.hyperliquid.xyz/exchange', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await res.json();

    const statusErr = result?.response?.data?.statuses?.[0]?.error;
    if (statusErr) {
      return Response.json({ ok: false, error: statusErr, result }, { status: 400 });
    }
    if (result?.status === 'err') {
      return Response.json({ ok: false, error: result.response, result }, { status: 400 });
    }

    return Response.json({ ok: true, result, signer: wallet.address });
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
});