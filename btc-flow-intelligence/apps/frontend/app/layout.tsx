import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BTC Flow Intelligence",
  description:
    "Real-time Bitcoin accumulation/distribution intelligence across ETF flows, on-chain, derivatives, entities and sentiment.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Loaded at runtime (not build time) so the production build never
            depends on network access to Google Fonts. The CSS variables
            --font-sans / --font-mono are defined in globals.css. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen antialiased">
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
