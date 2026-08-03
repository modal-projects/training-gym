// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { modalReferenceThemes } from './modal-reference-theme.mjs';
import { rehypeTableWrapper } from './rehype-table-wrapper.mjs';

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeTableWrapper],
  },
  site: 'https://gym.modal.dev',
  redirects: {
    '/tutorials/tools/000_observability_dashboard':
      '/guides/tools/observability-dashboard/',
    '/tutorials/tools/001_wandb_integration': '/guides/tools/wandb-integration/',
  },
  integrations: [
    starlight({
      title: 'Training Gym',
      favicon: '/modal-logo.svg',
      description:
        'Open-source Python SDK for GRPO and RL post-training of LLMs on Modal GPU clusters — tutorials, API reference, and runnable examples.',
      tagline:
        'GRPO, PPO, custom reward and generate functions — runnable on Modal in minutes.',
      head: [
        {
          tag: 'meta',
          attrs: { property: 'og:image', content: 'https://gym.modal.dev/og-image.png' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:width', content: '2400' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:height', content: '1260' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:card', content: 'summary_large_image' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:image', content: 'https://gym.modal.dev/og-image.png' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:site', content: '@modal_labs' },
        },
        {
          tag: 'script',
          attrs: { type: 'application/ld+json' },
          content: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: 'Training Gym',
            url: 'https://gym.modal.dev',
            description:
              'Open-source Python SDK for GRPO and RL post-training of LLMs on Modal GPU clusters.',
            publisher: {
              '@type': 'Organization',
              name: 'Modal Labs',
              url: 'https://modal.com',
            },
          }),
        },
      ],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/modal-projects/training-gym',
        },
      ],
      expressiveCode: {
        themes: modalReferenceThemes,
        useStarlightUiThemeColors: true,
        customizeTheme: (theme) => {
          theme.bg = '#1c1c1c';
          theme.fg = '#d1d1d1';
          theme.colors['editor.background'] = '#1c1c1c';
          theme.colors['editor.foreground'] = '#d1d1d1';
          theme.colors['editor.selectionBackground'] = '#2f2f2f';
          theme.colors['scrollbarSlider.background'] = '#ffffff17';
          theme.colors['scrollbarSlider.hoverBackground'] = '#ffffff40';
          theme.colors['titleBar.activeBackground'] = '#181818';
          theme.colors['titleBar.activeForeground'] = '#d1d1d1';
          theme.colors['titleBar.border'] = '#2f2f2f';
          theme.colors['editorGroupHeader.tabsBackground'] = '#181818';
          theme.colors['editorGroupHeader.tabsBorder'] = '#2f2f2f';
          theme.colors['tab.activeBackground'] = '#1c1c1c';
          theme.colors['tab.activeForeground'] = '#e8e8e8';
          theme.colors['tab.inactiveBackground'] = '#181818';
          theme.colors['tab.inactiveForeground'] = '#a3a3a3';
          theme.colors['tab.activeBorderTop'] = '#7fee64';
          theme.styleOverrides.frames = {
            ...theme.styleOverrides.frames,
            editorBackground: '#1c1c1c',
            terminalBackground: '#1c1c1c',
            editorActiveTabBackground: '#1c1c1c',
            inlineButtonForeground: '#d1d1d1',
            frameBoxShadowCssValue: 'none',
          };
          return theme;
        },
        styleOverrides: {
          borderRadius: '0.375rem',
          codeBackground: '#1c1c1c',
          codeForeground: '#d1d1d1',
          codeSelectionBackground: '#2f2f2f',
          gutterBorderColor: '#2f2f2f',
          gutterForeground: '#747474',
          gutterHighlightForeground: '#e8e8e8',
        },
      },
      customCss: ['./src/styles/custom.css'],
      components: {
        Header: './src/components/Header.astro',
        Search: './src/components/Search.astro',
        Sidebar: './src/components/Sidebar.astro',
        PageSidebar: './src/components/PageSidebar.astro',
        PageTitle: './src/components/PageTitle.astro',
      },
      sidebar: [
        { label: 'Overview', link: '/' },
        {
          label: 'Guides',
          items: [
            { label: 'Overview', link: '/guides/' },
            {
              label: 'Tools',
              autogenerate: { directory: 'guides/tools' },
            },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            { label: 'All Tutorials', link: '/tutorials/' },
            {
              label: 'Reinforcement Learning',
              autogenerate: { directory: 'tutorials/rl' },
            },
            {
              label: 'Agents',
              autogenerate: { directory: 'tutorials/agent' },
            },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Overview', link: '/reference/' },
            { label: 'CLI Reference', link: '/reference/cli/' },
            {
              label: 'Core',
              items: [
                { label: 'ModelConfig', link: '/reference/core/modelconfig/' },
                { label: 'HFModelConfiguration', link: '/reference/core/hfmodelconfiguration/' },
                { label: 'ModelArchitecture', link: '/reference/core/modelarchitecture/' },
                { label: 'DatasetConfig', link: '/reference/core/datasetconfig/' },
                { label: 'HuggingFaceDataset', link: '/reference/core/huggingfacedataset/' },
                { label: 'HarborDataset', link: '/reference/core/harbordataset/' },
                { label: 'WandbConfig', link: '/reference/core/wandbconfig/' },
                { label: 'ModalRayCluster', link: '/reference/core/modalraycluster/' },
                { label: 'TrainResult', link: '/reference/core/trainresult/' },
              ],
            },
            {
              label: 'Models',
              items: [
                { label: 'ToolCall', link: '/reference/models/toolcall/' },
                { label: 'ParsedResponse', link: '/reference/models/parsedresponse/' },
                { label: 'parse_qwen3_response', link: '/reference/models/parse_qwen3_response/' },
                { label: 'Qwen3-0.6B', link: '/reference/models/qwen3_0_6b/' },
                { label: 'Qwen3-1.7B', link: '/reference/models/qwen3_1_7b/' },
                { label: 'Qwen3-4B', link: '/reference/models/qwen3_4b/' },
                { label: 'Qwen3-8B', link: '/reference/models/qwen3_8b/' },
                { label: 'Qwen3-30B-A3B', link: '/reference/models/qwen3_30b/' },
                { label: 'Qwen3.5-4B', link: '/reference/models/qwen3_5_4b/' },
                {
                  label: 'Moonlight-16B-A3B-Instruct',
                  link: '/reference/models/moonlight_16b_a3b_instruct/',
                },
                { label: 'Qwen3.6-35B-A3B', link: '/reference/models/qwen3_6_35b/' },
                { label: 'Qwen3.6-27B', link: '/reference/models/qwen3_6_27b/' },
                { label: 'Qwen3.8-27B', link: '/reference/models/qwen3_8_27b/' },
              ],
            },
            {
              label: 'Training',
              items: [
                { label: 'TrainConfig', link: '/reference/training/trainconfig/' },
                { label: 'SlimeRecipe', link: '/reference/training/slimerecipe/' },
                { label: 'MilesRecipe', link: '/reference/training/milesrecipe/' },
                {
                  label: 'Qwen3_5_4b_Miles_Recipe',
                  link: '/reference/training/qwen3_5_4b_miles_recipe/',
                },
                {
                  label: 'Moonlight_16B_A3B_Recipe',
                  link: '/reference/training/moonlight_16b_a3b_recipe/',
                },
                { label: 'Qwen3_6_35b_Recipe', link: '/reference/training/qwen3_6_35b_recipe/' },
                { label: 'Qwen3_6_27b_Recipe', link: '/reference/training/qwen3_6_27b_recipe/' },
                { label: 'Qwen3_8_27b_Recipe', link: '/reference/training/qwen3_8_27b_recipe/' },
              ],
            },
            {
              label: 'Deployment',
              items: [
                { label: 'Endpoint', link: '/reference/deployment/endpoint/' },
                { label: 'CustomDeployment', link: '/reference/deployment/customdeployment/' },
                { label: 'SglangRecipe', link: '/reference/deployment/sglangrecipe/' },
                { label: 'VllmRecipe', link: '/reference/deployment/vllmrecipe/' },
              ],
            },
          ],
        },
        { label: 'CLI Reference', link: '/reference/cli/' },
        { label: 'Support', link: '/support/' },
      ],
      lastUpdated: true,
      disable404Route: true,
    }),
  ],
});
