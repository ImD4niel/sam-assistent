/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        chatgpt: {
          sidebar: '#000000',
          main: '#212121',
          input: '#2F2F2F',
          bubble: '#2F2F2F',
        }
      }
    },
  },
  plugins: [],
}
