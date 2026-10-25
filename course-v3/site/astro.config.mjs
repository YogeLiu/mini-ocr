import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import fs from 'node:fs';
import rehypeCallouts from './src/plugins/rehype-callouts.mjs';

const sidebar = JSON.parse(fs.readFileSync(new URL('./sidebar.json', import.meta.url), 'utf8'));

export default defineConfig({
  site: 'https://yogeliu.github.io',
  base: process.env.SITE_BASE ?? '/mini-ocr',
  markdown: { smartypants: false, rehypePlugins: [rehypeCallouts] },
  integrations: [
    starlight({
      title: '从像素到 Xiaomi-OCR-0',
      description: '系统学习视觉语言模型与 OCR 的自学教程',
      defaultLocale: 'root',
      locales: { root: { label: '简体中文', lang: 'zh-CN' } },
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/YogeLiu/mini-ocr' }],
      sidebar,
      components: { PageTitle: './src/components/PageTitle.astro' },
      customCss: ['./src/styles/reading.css'],
      pagination: true,
      head: [{ tag: 'script', content: "addEventListener('DOMContentLoaded',()=>{const b=document.createElement('div');b.id='read-progress';document.body.appendChild(b);const u=()=>{const h=document.documentElement;b.style.width=(h.scrollTop/Math.max(1,h.scrollHeight-h.clientHeight)*100)+'%'};addEventListener('scroll',u,{passive:true});u()})" }],
      expressiveCode: { themes: ['github-dark-dimmed', 'github-light'], styleOverrides: { borderRadius: '6px' } },
    }),
  ],
});
