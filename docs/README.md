# XN docs

The documentation site for XN, built with [Astro](https://astro.build/) and [Starlight](https://starlight.astro.build/). It is published at https://apps.nknighta.me/xn/ by `.github/workflows/docs.yml` on every push to `main` that touches `docs/` or `install.sh`.

## Edit a page

Pages are MDX files in `src/content/docs/`. The file path is the URL:

| File | URL |
|---|---|
| `src/content/docs/index.mdx` | `/xn/` |
| `src/content/docs/getting-started.mdx` | `/xn/getting-started/` |
| `src/content/docs/guides/coding-model.mdx` | `/xn/guides/coding-model/` |

Every page starts with frontmatter:

```mdx
---
title: My page
description: One sentence shown in search results.
sidebar:
  order: 1   # optional: position within its sidebar group
---
```

To add a page, create a new `.mdx` file. Files in `guides/` and `reference/` appear in the sidebar automatically; top-level pages are listed in `astro.config.mjs`.

MDX is Markdown plus components. Import Starlight's built-in components at the top of the file:

```mdx
import { Steps, Tabs, TabItem, Card, CardGrid } from '@astrojs/starlight/components';

<Tabs>
  <TabItem label="npm">npm install</TabItem>
  <TabItem label="bun">bun install</TabItem>
</Tabs>
```

Callouts use `:::note`, `:::tip`, `:::caution` or `:::danger`. See the [Starlight components guide](https://starlight.astro.build/components/using-components/) for the rest.

### Languages

English pages are at the top of `src/content/docs/`; their Japanese versions are at the same path under `src/content/docs/ja/` (for example `guides/coding-model.mdx` and `ja/guides/coding-model.mdx`). When you add or change an English page, update the Japanese one too. A page without a translation falls back to English with a notice. Sidebar group labels are translated in `astro.config.mjs` (`translations: { ja: ... }`).

### Links

The site lives under `/xn/`, so link between pages with relative paths, such as `[Commands](../reference/commands/)`, not `/reference/commands/`.

## Preview

```sh
cd docs
npm install
npm run dev       # http://localhost:4321/xn/ with live reload
npm run build     # build into dist/
npm run preview   # serve the built site
```

`install.sh` is not edited here: `npm run dev` and `npm run build` copy it from the repository root into `public/`, so the site always serves the current script at `/xn/install.sh`.
