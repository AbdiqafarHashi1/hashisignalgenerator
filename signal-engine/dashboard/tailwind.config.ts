import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "binance-dark": "#0b0e11",
        "binance-card": "#151a21",
        "binance-border": "#1f2630",
        "binance-accent": "#f0b90b",
      },
      boxShadow: {
        panel: "0 0 0 1px rgba(255,255,255,0.04)",
      },
    },
  },
  plugins: [],
};

export default config;
