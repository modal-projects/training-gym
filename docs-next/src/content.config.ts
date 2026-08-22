import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsSchema } from '@astrojs/starlight/schema';
import { tutorialDocsLoader } from './lib/tutorial-docs-loader';

export const collections = {
  docs: defineCollection({
    loader: tutorialDocsLoader(),
    schema: docsSchema({
      extend: z.object({
        runCommand: z.string().optional(),
      }),
    }),
  }),
};
