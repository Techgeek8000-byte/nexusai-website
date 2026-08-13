import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* No output: "standalone" — this breaks Vercel builds */
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
