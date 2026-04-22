// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://training-gym.modal.dev',
  integrations: [
    starlight({
      title: 'Training Gym',
      description:
        'Reusable building blocks + runnable examples for distributed training on Modal.',
      tagline:
        'Framework-aware launchers, paired notebooks, and fewer one-off cluster scripts.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/modal-projects/training-gym',
        },
      ],
      customCss: ['./src/styles/custom.css'],
      components: {
        Header: './src/components/Header.astro',
        PageSidebar: './src/components/PageSidebar.astro',
        PageTitle: './src/components/PageTitle.astro',
      },
      sidebar: [
        { label: 'Overview', link: '/' },
        { label: 'Tutorials', link: '/tutorials/' },
        { label: 'Support', link: '/support/' },
      ],
      editLink: {
        baseUrl:
          'https://github.com/modal-projects/training-gym/edit/main/docs-next/src/content/docs/',
      },
      lastUpdated: true,
      disable404Route: true,
    }),
  ],
});
