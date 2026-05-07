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
              label: 'Reinforcement Learning',
              items: [
                { label: 'RL Basics', link: '/tutorials/rl/000_rl_basics/' },
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
                { label: 'ModelConfig', link: '/reference/core/modelconfig/' },
                { label: 'HFModelConfiguration', link: '/reference/core/hfmodelconfiguration/' },
                { label: 'ModelArchitecture', link: '/reference/core/modelarchitecture/' },
                { label: 'DatasetConfig', link: '/reference/core/datasetconfig/' },
                { label: 'HuggingFaceDataset', link: '/reference/core/huggingfacedataset/' },
                { label: 'WandbConfig', link: '/reference/core/wandbconfig/' },
                { label: 'ModalRayCluster', link: '/reference/core/modalraycluster/' },
                { label: 'TrainResult', link: '/reference/core/trainresult/' },
              ],
            },
            {
              label: 'Evaluation',
              items: [
                { label: 'EvalConfig', link: '/reference/evaluation/evalconfig/' },
                { label: 'EvalResult', link: '/reference/evaluation/evalresult/' },
                { label: 'EvalRowResult', link: '/reference/evaluation/evalrowresult/' },
              ],
            },
            {
              label: 'Models',
              items: [
                { label: 'Qwen3-0.6B', link: '/reference/models/qwen3_0_6b/' },
                { label: 'Qwen3-4B', link: '/reference/models/qwen3_4b/' },
                { label: 'Qwen3-30B-A3B', link: '/reference/models/qwen3_30b/' },
              ],
            },
            {
              label: 'Training',
              items: [
                { label: 'TrainConfig', link: '/reference/training/trainconfig/' },
                { label: 'SlimeRecipe', link: '/reference/training/slimerecipe/' },
              ],
            },
            {
              label: 'Deployment',
              items: [
                { label: 'DeploymentConfig', link: '/reference/deployment/deploymentconfig/' },
                { label: 'ModelDeployment', link: '/reference/deployment/modeldeployment/' },
                { label: 'SglangRecipe', link: '/reference/deployment/sglangrecipe/' },
                { label: 'VllmRecipe', link: '/reference/deployment/vllmrecipe/' },
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
