// @ts-check
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { rehypeTableWrapper } from './rehype-table-wrapper.mjs';
import { parseTutorialMetadata } from './src/lib/tutorial-docs-loader.ts';

const configDirectory = path.dirname(fileURLToPath(import.meta.url));

function remarkStripPageTitle() {
  return (/** @type {{ children: Array<{ type: string, depth?: number }> }} */ tree) => {
    const index = tree.children.findIndex(
      (node) => node.type === 'heading' && node.depth === 1,
    );
    if (index !== -1) {
      tree.children.splice(index, 1);
    }
  };
}

function loadTutorialPages() {
  const tutorialsDirectory = path.resolve(configDirectory, '../tutorials');
  const tutorials = readdirSync(tutorialsDirectory)
    .filter((fileName) => fileName.endsWith('.py'))
    .map((fileName) => {
      const tutorialPath = path.join(tutorialsDirectory, fileName);
      const source = readFileSync(tutorialPath, 'utf8');
      const { order } = parseTutorialMetadata(source, tutorialPath);
      return { order, slug: path.basename(fileName, '.py') };
    })
    .sort((left, right) => left.order - right.order || left.slug.localeCompare(right.slug));
  if (!tutorials[0]) {
    throw new Error('No tutorial pages found');
  }
  return tutorials;
}

function firstGuidePath() {
  const guidesDirectory = path.resolve(configDirectory, 'src/content/docs/guides');
  const guides = readdirSync(guidesDirectory, { recursive: true, encoding: 'utf8' })
    .filter((fileName) => fileName.endsWith('.md'))
    .map((fileName) => {
      const guidePath = path.join(guidesDirectory, fileName);
      const source = readFileSync(guidePath, 'utf8');
      const frontmatterEnd = source.indexOf('\n---', 4);
      const frontmatter = frontmatterEnd === -1 ? '' : source.slice(4, frontmatterEnd);
      const orderMatch = frontmatter.match(/^order:\s*(\d+)\s*$/m);
      if (!orderMatch) {
        throw new Error(`${guidePath} is missing order`);
      }
      const slug = fileName
        .replaceAll(path.sep, '/')
        .replace(/(?:\/index)?\.md$/, '');
      return { order: Number(orderMatch[1]), slug };
    })
    .sort((left, right) => left.order - right.order || left.slug.localeCompare(right.slug));
  if (!guides[0]) {
    throw new Error('No guide pages found');
  }
  return `/guides/${guides[0].slug}`;
}

const tutorialPages = loadTutorialPages();
const firstTutorial = `/tutorials/${tutorialPages[0].slug}`;
const firstGuide = firstGuidePath();

export default defineConfig({
  trailingSlash: 'never',
  redirects: {
    '/guides': firstGuide,
    '/support': '/',
    '/tutorials': firstTutorial,
    '/tutorials/agent': firstTutorial,
    '/tutorials/agent/000_agent_sandbox': '/tutorials/agent_sandbox',
    '/tutorials/multinode/002_glm_4_7': '/tutorials/multinode',
    '/tutorials/rl': firstTutorial,
    '/tutorials/rl/000_rl_basics': '/tutorials/rl_basics',
    '/tutorials/rl/001_sandboxes': '/tutorials/sandboxes',
    '/tutorials/rl/002_multiturn': '/tutorials/multiturn',
    '/tutorials/rl/003_on_policy_distillation': '/tutorials/on_policy_distillation',
    '/tutorials/rl/005_dapo': '/tutorials/dapo',
    '/tutorials/rl/006_audio_asr': '/tutorials/audio_asr',
    '/tutorials/rl/007_param_sweep': '/tutorials/param_sweep',
    '/tutorials/rl/008_computer_use': '/tutorials/computer_use',
    '/tutorials/rl/009_cross_tokenizer_distillation':
      '/tutorials/cross_tokenizer_distillation',
    '/tutorials/tools/000_observability_dashboard':
      '/guides/tools/observability-dashboard',
    '/tutorials/tools/001_wandb_integration': '/guides/tools/wandb-integration',
    '/tutorials/rl/010_agent_driven_training': '/guides/agent-driven-training',
  },
  markdown: {
    remarkPlugins: [remarkMath, remarkStripPageTitle],
    rehypePlugins: [rehypeKatex, rehypeTableWrapper],
  },
  site: 'https://gym.modal.dev',
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
        TableOfContents: './src/components/TableOfContents.astro',
      },
      sidebar: [
        {
          label: 'Guides',
          items: [
            { label: 'Agent-driven training', link: '/guides/agent-driven-training' },
            {
              label: 'Tools',
              autogenerate: { directory: 'guides/tools' },
            },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            {
              label: 'Featured',
              items: tutorialPages.map(({ slug }) => ({ slug: `tutorials/${slug}` })),
            },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Overview', link: '/reference' },
            { label: 'CLI Reference', link: '/reference/cli' },
            {
              label: 'Core',
              items: [
                { label: 'ModelConfig', link: '/reference/core/modelconfig' },
                { label: 'HFModelConfiguration', link: '/reference/core/hfmodelconfiguration' },
                { label: 'ModelArchitecture', link: '/reference/core/modelarchitecture' },
                { label: 'DatasetConfig', link: '/reference/core/datasetconfig' },
                { label: 'HuggingFaceDataset', link: '/reference/core/huggingfacedataset' },
                { label: 'HarborDataset', link: '/reference/core/harbordataset' },
                { label: 'MetricConfig', link: '/reference/core/metricconfig' },
                { label: 'WandbConfig', link: '/reference/core/wandbconfig' },
                { label: 'ModalRayCluster', link: '/reference/core/modalraycluster' },
                { label: 'TrainResult', link: '/reference/core/trainresult' },
              ],
            },
            {
              label: 'Models',
              items: [
                { label: 'ToolCall', link: '/reference/models/toolcall' },
                { label: 'ParsedResponse', link: '/reference/models/parsedresponse' },
                { label: 'parse_qwen3_response', link: '/reference/models/parse_qwen3_response' },
                { label: 'Qwen3-0.6B', link: '/reference/models/qwen3_0_6b' },
                { label: 'Qwen3-1.7B', link: '/reference/models/qwen3_1_7b' },
                { label: 'Qwen3-4B', link: '/reference/models/qwen3_4b' },
                { label: 'Qwen3-8B', link: '/reference/models/qwen3_8b' },
                { label: 'Qwen3-30B-A3B', link: '/reference/models/qwen3_30b' },
                { label: 'Qwen3.5-4B', link: '/reference/models/qwen3_5_4b' },
                {
                  label: 'Moonlight-16B-A3B-Instruct',
                  link: '/reference/models/moonlight_16b_a3b_instruct',
                },
                { label: 'Qwen3.6-35B-A3B', link: '/reference/models/qwen3_6_35b' },
                { label: 'Qwen3.6-27B', link: '/reference/models/qwen3_6_27b' },
                { label: 'Qwen3.8-27B', link: '/reference/models/qwen3_8_27b' },
              ],
            },
            {
              label: 'Training',
              items: [
                { label: 'TrainConfig', link: '/reference/training/trainconfig' },
                { label: 'SlimeRecipe', link: '/reference/training/slimerecipe' },
                { label: 'MilesRecipe', link: '/reference/training/milesrecipe' },
                {
                  label: 'Qwen3_5_4b_Miles_Recipe',
                  link: '/reference/training/qwen3_5_4b_miles_recipe',
                },
                {
                  label: 'Moonlight_16B_A3B_Recipe',
                  link: '/reference/training/moonlight_16b_a3b_recipe',
                },
                { label: 'Qwen3_6_35b_Recipe', link: '/reference/training/qwen3_6_35b_recipe' },
                { label: 'Qwen3_6_27b_Recipe', link: '/reference/training/qwen3_6_27b_recipe' },
                { label: 'Qwen3_8_27b_Recipe', link: '/reference/training/qwen3_8_27b_recipe' },
              ],
            },
            {
              label: 'Deployment',
              items: [
                { label: 'Endpoint', link: '/reference/deployment/endpoint' },
                { label: 'CustomDeployment', link: '/reference/deployment/customdeployment' },
                { label: 'SglangRecipe', link: '/reference/deployment/sglangrecipe' },
                { label: 'VllmRecipe', link: '/reference/deployment/vllmrecipe' },
              ],
            },
          ],
        },
        { label: 'CLI Reference', link: '/reference/cli' },
      ],
      lastUpdated: false,
      pagination: false,
      disable404Route: true,
    }),
  ],
});
