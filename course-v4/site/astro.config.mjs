import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import fs from 'node:fs';
import rehypeCallouts from './src/plugins/rehype-callouts.mjs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const sidebar = JSON.parse(fs.readFileSync(new URL('./sidebar.json', import.meta.url), 'utf8'));

export default defineConfig({
  site: 'https://yogeliu.github.io',
  base: process.env.SITE_BASE ?? '/mini-ocr',
  markdown: {
    smartypants: false,
    remarkPlugins: [[remarkMath, { singleDollarTextMath: false }]],
    rehypePlugins: [rehypeKatex, rehypeCallouts],
  },
  integrations: [
    starlight({
      title: 'Mini OCR：从零到公式 OCR',
      description: '从自动求导到微调视觉语言模型的中文教材',
      defaultLocale: 'root',
      locales: { root: { label: '简体中文', lang: 'zh-CN' } },
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/YogeLiu/mini-ocr' }],
      sidebar,
      components: {
        PageTitle: './src/components/PageTitle.astro',
        Header: './src/components/Header.astro',
      },
      customCss: ['katex/dist/katex.min.css', './src/styles/reading.css'],
      pagination: true,
      head: [{ tag: 'script', content: "addEventListener('DOMContentLoaded',()=>{const b=document.createElement('div');b.id='read-progress';document.body.appendChild(b);const u=()=>{const h=document.documentElement;b.style.width=(h.scrollTop/Math.max(1,h.scrollHeight-h.clientHeight)*100)+'%'};addEventListener('scroll',u,{passive:true});u()})" }],
      expressiveCode: { themes: ['github-dark-dimmed', 'github-light'], styleOverrides: { borderRadius: '6px' } },
    }),
  ],
});
