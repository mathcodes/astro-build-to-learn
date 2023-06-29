import {defineConfig} from 'astro/config';
import installedIntegration from '@astrojs/react';

export default defineConfig({
  integrations: [
    // 1. Imported from an installed npm package
    installedIntegration(),
  ]
})