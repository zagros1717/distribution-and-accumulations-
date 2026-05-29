/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    // Proxy /api/* to the backend so the browser never needs CORS in prod.
    // BACKEND_URL remains overridable in Railway Variables; the production
    // fallback keeps the frontend functional when that variable is missing.
    const backend = process.env.BACKEND_URL || "https://distribution-and-accumulations-production.up.railway.app";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};
module.exports = nextConfig;
