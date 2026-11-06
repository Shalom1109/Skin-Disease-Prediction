// Vercel build step: writes frontend/js/config.js from the SKINSENSE_API_URL environment variable.
// The front end is static; the API (PyTorch, model weights, database) must run on another host.
import { writeFileSync } from 'node:fs';

const raw = (process.env.SKINSENSE_API_URL || '').trim().replace(/\/+$/, '');

// Only a plain http(s) URL may reach the generated script: no quotes, spaces, angle brackets or backslashes.
const PLAIN_URL = /^https?:\/\/[^\s'"<>\\]+$/;

if (raw && !PLAIN_URL.test(raw)) {
    console.error('SKINSENSE_API_URL must be a plain http(s) URL, e.g. https://skinsense-api.example.com');
    process.exit(1);
}
if (raw.startsWith('http://') && !/^http:\/\/(localhost|127\.0\.0\.1)/.test(raw)) {
    console.warn('WARNING: an http:// API cannot be called from an https:// Vercel site (mixed content). Use https.');
}

writeFileSync('frontend/js/config.js', `window.SKINSENSE_API_URL = ${JSON.stringify(raw)};\n`);
console.log(raw ? `API URL set to ${raw}` : 'SKINSENSE_API_URL is not set: the site will look for the API on its own address.');
