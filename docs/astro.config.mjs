// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Served by GitHub Pages at https://apps.nknighta.me/xn/
export default defineConfig({
  site: 'https://apps.nknighta.me',
  base: '/xn',
  // let the dev server answer through the Cloudflare tunnel (see README.md)
  server: { allowedHosts: ['pctunnel.nknighta.me'] },
  integrations: [
    starlight({
      title: 'XN',
      description: 'Eazy make a LLM. Train small GPT-style language models from scratch on a CPU.',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/nk4dev/xn' }],
      // English at /xn/, Japanese at /xn/ja/ (pages in src/content/docs/ja/)
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        ja: { label: '日本語', lang: 'ja' },
      },
      editLink: { baseUrl: 'https://github.com/nk4dev/xn/edit/main/docs/' },
      sidebar: [
        {
          label: 'Start here',
          translations: { ja: 'はじめに' },
          items: ['getting-started', 'how-it-works'],
        },
        {
          label: 'Guides',
          translations: { ja: 'ガイド' },
          items: [{ autogenerate: { directory: 'guides' } }],
        },
        {
          label: 'Reference',
          translations: { ja: 'リファレンス' },
          items: [{ autogenerate: { directory: 'reference' } }],
        },
      ],
    }),
  ],
});
