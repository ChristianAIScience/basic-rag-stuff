// Docusaurus configuration for the RAG Learning Lab documentation website.
// The pages are NOT copied: the site reads the Markdown files in ../docs directly,
// so the explanations exist only once and stay in sync with the project.

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'RAG Learning Lab',
  tagline: 'Learn Retrieval-Augmented Generation with LangChain, LangGraph and PostgreSQL/pgvector',
  url: 'http://localhost',
  baseUrl: '/',

  onBrokenLinks: 'throw',

  // Languages: English (default, from ../docs) and German (translations in i18n/de/).
  // German pages live under /de/... - the language switch is in the navbar.
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'de'],
    localeConfigs: {
      en: {label: 'English'},
      de: {label: 'Deutsch'},
    },
  },

  markdown: {
    // .md files are parsed as plain CommonMark (not MDX), so text like {"a": 1} needs no escaping.
    format: 'detect',
    mermaid: true, // render ```mermaid code blocks as diagrams
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../docs', // the project's documentation folder
          sidebarPath: './sidebars.js',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'RAG Learning Lab',
        items: [
          {type: 'docSidebar', sidebarId: 'learningPath', position: 'left', label: 'Learning path'},
          {type: 'localeDropdown', position: 'right'},
        ],
      },
      footer: {
        style: 'dark',
        copyright: 'RAG Learning Lab - a learning project. Sample data describes the fictional company Lumora Systems.',
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'powershell', 'sql', 'toml', 'batch'],
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
    }),
};

export default config;
