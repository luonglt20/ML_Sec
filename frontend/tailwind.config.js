/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0B0F17',
          800: '#0F172A',
          700: '#1E293B',
          600: '#334155',
        },
        brand: {
          cyan: '#38BDF8',
          indigo: '#818CF8',
          emerald: '#34D399',
          purple: '#C084FC',
        }
      }
    },
  },
  plugins: [],
}
