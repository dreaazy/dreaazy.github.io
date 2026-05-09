// src/content.config.ts
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const projects = defineCollection({
  loader: glob({ base: './src/content/projects', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(), // Made optional for sub-files
    pubDate: z.coerce.date(),
    order: z.number().default(0), // Added for sequencing
    heroImage: z.string().optional(),
  }),
});


const notes = defineCollection({
  loader: glob({ base: './src/content/notes', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).optional(), // Handy for Obsidian tags!
  }),
});

export const collections = { projects, notes };