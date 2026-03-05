/** @type {import('next').NextConfig} */
const nextConfig = {
    // Move 'turbopack' out of 'experimental' and into the root
    turbopack: {}, 
    
    webpack: (config, { isServer }) => {
      if (!isServer) {
        config.resolve.fallback = {
          fs: false,
          net: false,
          tls: false,
          path: false,
          os: false,
        };
      }
      return config;
    },
  };
  
  export default nextConfig;