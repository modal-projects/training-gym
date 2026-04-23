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
        {
          label: 'API Reference',
          items: [
            { label: 'Overview', link: '/reference/' },
            {
              label: 'Core',
              items: [
                { label: 'ModelConfiguration', link: '/reference/core/modelconfiguration/' },
                { label: 'HFModelConfiguration', link: '/reference/core/hfmodelconfiguration/' },
                { label: 'ModelArchitecture', link: '/reference/core/modelarchitecture/' },
                { label: 'DatasetConfig', link: '/reference/core/datasetconfig/' },
                { label: 'WandbConfig', link: '/reference/core/wandbconfig/' },
                { label: 'ModalRayCluster', link: '/reference/core/modalraycluster/' },
                { label: 'LlmJudge', link: '/reference/core/llmjudge/' },
              ],
            },
            {
              label: 'Models',
              items: [
                { label: 'Qwen3-4B', link: '/reference/models/qwen3_4b/' },
                { label: 'Qwen3-32B', link: '/reference/models/qwen3_32b/' },
                { label: 'GLM-4.7', link: '/reference/models/glm_4_7/' },
                { label: 'Llama2-7B', link: '/reference/models/llama2_7b/' },
                { label: 'Kimi-K2.5', link: '/reference/models/kimi_k2_5/' },
              ],
            },
            {
              label: 'Frameworks',
              items: [
                { label: 'SlimeConfig', link: '/reference/frameworks/slimeconfig/' },
                { label: 'ModalConfig (SLIME)', link: '/reference/frameworks/modalconfig/' },
                { label: 'MsSwiftFrameworkConfig', link: '/reference/frameworks/msswiftframeworkconfig/' },
                { label: 'MsSwiftConfig', link: '/reference/frameworks/msswiftconfig/' },
                { label: 'MilesFrameworkConfig', link: '/reference/frameworks/milesframeworkconfig/' },
                { label: 'MilesConfig', link: '/reference/frameworks/milesconfig/' },
                { label: 'HarborFrameworkConfig', link: '/reference/frameworks/harborframeworkconfig/' },
                { label: 'HarborConfig', link: '/reference/frameworks/harborconfig/' },
              ],
            },
          ],
        },
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
