// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://training-gym-next.modal.dev',
  integrations: [
    starlight({
      title: 'Training Gym',
      description:
        'Reusable building blocks + runnable examples for distributed training on Modal.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/modal-projects/training-gym',
        },
      ],
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        { label: 'Overview', link: '/' },
        {
          label: 'Tutorials',
          autogenerate: { directory: 'tutorials' },
        },
        { label: 'Support', link: '/support/' },
      ],
      editLink: {
        baseUrl:
          'https://github.com/modal-projects/training-gym/edit/main/docs-next/src/content/docs/',
      },
      lastUpdated: true,
    }),
  ],
});
