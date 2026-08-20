// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
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
      // Code blocks mirror the main Modal docs site (modal.com/docs): shiki's
      // dark-plus theme, Fira Mono 14px/20px, and a frameless container with a
      // translucent white surface over the page background.
      expressiveCode: {
        themes: ['dark-plus'],
        useStarlightUiThemeColors: false,
        defaultProps: { frame: 'none' },
        // The code surface is translucent white, which Expressive Code's
        // contrast pass reads as a light background and compensates for by
        // darkening every token color. Off, so dark-plus renders as-is.
        minSyntaxHighlightingColorContrast: 0,
        styleOverrides: {
          borderColor: '#2f2f2f',
          borderRadius: '0.25rem',
          borderWidth: '1px',
          codeBackground: 'rgba(255, 255, 255, 0.06)',
          codeForeground: '#d4d4d4',
          codeFontFamily: 'var(--sl-font-mono)',
          codeFontSize: '0.875rem',
          codeLineHeight: '1.4286',
          codePaddingBlock: '0.875rem',
          codePaddingInline: '0.875rem',
          codeSelectionBackground: '#264f78',
          scrollbarThumbColor: '#ffffff17',
          scrollbarThumbHoverColor: '#ffffff40',
          gutterBorderColor: '#2f2f2f',
          gutterForeground: '#747474',
          gutterHighlightForeground: '#e8e8e8',
          frames: {
            frameBoxShadowCssValue: 'none',
            editorBackground: 'rgba(255, 255, 255, 0.06)',
            terminalBackground: 'rgba(255, 255, 255, 0.06)',
            inlineButtonBackground: '#222222',
            inlineButtonBackgroundIdleOpacity: '1',
            inlineButtonBackgroundHoverOrFocusOpacity: '1',
            inlineButtonBackgroundActiveOpacity: '1',
            inlineButtonBorder: '#464646',
            inlineButtonBorderOpacity: '1',
            inlineButtonForeground: '#d1d1d1',
            tooltipSuccessBackground: '#222222',
            tooltipSuccessForeground: '#d1d1d1',
          },
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
                { label: 'MetricConfig', link: '/reference/core/metricconfig/' },
                { label: 'WandbConfig', link: '/reference/core/wandbconfig/' },
                { label: 'TrackioConfig', link: '/reference/core/trackioconfig/' },
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
