import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produce the minimal Node.js server used by the production container.
  output: "standalone",
};

export default nextConfig;
