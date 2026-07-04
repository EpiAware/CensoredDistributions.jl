// MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
// The standard DocumenterVitepress theme: the default theme plus the version
// picker, tab support, and the Nolebase enhanced-readabilities menu.
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'

import {
  NolebaseEnhancedReadabilitiesMenu,
  NolebaseEnhancedReadabilitiesScreenMenu,
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
// GitHub-stars navbar widget (live star count + repo link). Mirrors
// CensoredDistributions.jl.
import StarUs from "@/StarUs.vue"

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css'
// DocumenterVitepress writes `docstrings.css` next to this file at build time;
// importing it styles the API reference docstring blocks (matches
// CensoredDistributions.jl). Without it the `@docs` blocks render unstyled.
import './docstrings.css'

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => [
        h(StarUs),
        h(NolebaseEnhancedReadabilitiesMenu),
      ],
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
  }
}
export default Theme
