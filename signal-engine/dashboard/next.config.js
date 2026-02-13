/** @type {import('next').NextConfig} */
const apiOrigin = process.env.API_ORIGIN || 'http://api:8000';

const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${apiOrigin}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
