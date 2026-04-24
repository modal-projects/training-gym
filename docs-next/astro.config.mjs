// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { modalReferenceThemes } from './modal-reference-theme.mjs';

export default defineConfig({
  site: 'https://gym.modal.dev',
  integrations: [
    starlight({
      title: 'Training Gym',
      favicon: '/modal-logo.svg',
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
        Sidebar: './src/components/Sidebar.astro',
        PageSidebar: './src/components/PageSidebar.astro',
        PageTitle: './src/components/PageTitle.astro',
      },
      sidebar: [
        { label: 'Overview', link: '/' },
        {
          label: 'Tutorials',
          items: [
            { label: 'Overview', link: '/tutorials/' },
            {
              label: 'Getting Started',
              items: [
                { label: '001 Quickstart', link: '/tutorials/intro/001_quickstart/' },
                { label: '002 Custom Model', link: '/tutorials/intro/002_custom_model/' },
              ],
            },
            {
              label: 'Reinforcement Learning',
              items: [
                { label: '001 slime Intro', link: '/tutorials/rl/001_slime_intro/' },
                { label: '002 Customizing slime', link: '/tutorials/rl/002_customizing_your_slime_run/' },
                { label: '003 slime + LLM Judge', link: '/tutorials/rl/003_slime_with_llm_as_judge/' },
                { label: '004 Harbor Code Golf', link: '/tutorials/rl/004_harbor_codegolf/' },
              ],
            },
            {
              label: 'Supervised Fine-Tuning',
              items: [
                { label: '001 ms-swift', link: '/tutorials/sft/001_ms_swift/' },
              ],
            },
            {
              label: 'Infrastructure',
              items: [
                { label: 'Ray Standalone', link: '/tutorials/misc/ray_slime_standalone/' },
              ],
            },
          ],
        },
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
                { label: 'TrainResult', link: '/reference/core/trainresult/' },
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
                { label: 'ModalConfig (slime)', link: '/reference/frameworks/modalconfig/' },
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
          'https://github.com/modal-projects/training-gym/edit/joy/initial-setup/docs-next/src/content/docs/',
      },
      lastUpdated: true,
      disable404Route: true,
    }),
  ],
});
