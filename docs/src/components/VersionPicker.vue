<!-- Custom VersionPicker for custom domains - loads versions.js from root -->
<!-- Based on https://github.com/LuxDL/DocumenterVitepress.jl/blob/main/template/src/components/VersionPicker.vue -->

<script setup lang="ts">
import { ref, onMounted, computed} from 'vue'
import { useData } from 'vitepress'
import VPNavBarMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavBarMenuGroup.vue'
import VPNavScreenMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavScreenMenuGroup.vue'

declare global {
  interface Window {
    DOC_VERSIONS?: string[];
    DOCUMENTER_CURRENT_VERSION?: string;
    DOCUMENTER_STABLE?: string;
  }
}

function joinPath(base: string, path: string) {
  return `${base}${path}`.replace(/\/+/g, '/')
}

const absoluteRoot = __DEPLOY_ABSPATH__;
const siteOrigin = (typeof window === 'undefined' ? '' : window.location.origin);

function absoluteUrl(relative) {
  const withRoot = joinPath(absoluteRoot, relative);
  return siteOrigin + withRoot;
}

const props = defineProps<{ screenMenu?: boolean }>();
const versions = ref<Array<{ text: string, link: string, class?: string }>>([]);
const currentVersion = ref('Versions');
const isClient = ref(false);
const { site } = useData();

const isLocalBuild = () => {
  return typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
};

// Custom: Load versions.js from root for custom domains
const loadVersionsScript = () => {
  return new Promise<boolean>((resolve) => {
    if (typeof window === 'undefined') {
      resolve(false);
      return;
    }

    // If already loaded, resolve immediately
    if (window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
      resolve(true);
      return;
    }

    // Try to load from root path (for custom domains)
    const script = document.createElement('script');
    script.src = '/versions.js';
    script.onload = () => {
      // Give it a moment to execute
      setTimeout(() => {
        resolve(!!window.DOC_VERSIONS);
      }, 50);
    };
    script.onerror = () => {
      resolve(false);
    };
    document.head.appendChild(script);
  });
};

const waitForScriptsToLoad = () => {
  return new Promise<boolean>((resolve) => {
    if (isLocalBuild() || typeof window === 'undefined') {
      resolve(false);
      return;
    }

    // First check if already loaded
    if (window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
      resolve(true);
      return;
    }

    // Try loading from root
    loadVersionsScript().then((loaded) => {
      if (loaded) {
        resolve(true);
      } else {
        // Fall back to waiting for the original script
        const checkInterval = setInterval(() => {
          if (window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
            clearInterval(checkInterval);
            resolve(true);
          }
        }, 100);
        setTimeout(() => {
          clearInterval(checkInterval);
          resolve(false);
        }, 5000);
      }
    });
  });
};

const loadVersions = async () => {
  if (typeof window === 'undefined') return;

  try {
    if (isLocalBuild()) {
      versions.value = [{ text: 'dev', link: '/' }];
      currentVersion.value = 'dev';
    } else {
      const scriptsLoaded = await waitForScriptsToLoad();

      if (scriptsLoaded && window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
        versions.value = window.DOC_VERSIONS.map(v => ({
          text: v,
          link: absoluteUrl(`/${v}/`),
        }));
        currentVersion.value = window.DOCUMENTER_CURRENT_VERSION;
      } else {
        versions.value = [{ text: 'dev', link: absoluteUrl('/dev/') }];
        currentVersion.value = 'dev';
      }
    }
  } catch (error) {
    console.warn('Error loading versions:', error);
    versions.value = [{ text: 'dev', link: absoluteUrl('/dev/') }];
    currentVersion.value = 'dev';
  }
  isClient.value = true;
};

const versionItems = computed(() => {
  return versions.value.map((v) => ({
    text: v.text,
    link: v.link
  }));
});

onMounted(() => {
  if (typeof window !== 'undefined') {
    currentVersion.value = window.DOCUMENTER_CURRENT_VERSION ?? 'Versions';
    loadVersions();
  }
});
</script>

<template>
  <template v-if="isClient">
    <VPNavBarMenuGroup
      v-if="!screenMenu && versions.length > 0"
      :item="{ text: currentVersion, items: versionItems }"
      class="VPVersionPicker"
    />
    <VPNavScreenMenuGroup
      v-else-if="screenMenu && versions.length > 0"
      :text="currentVersion"
      :items="versionItems"
      class="VPVersionPicker"
    />
  </template>
</template>

<style scoped>
.VPVersionPicker :deep(button .text) {
  color: var(--vp-c-text-1) !important;
}
.VPVersionPicker:hover :deep(button .text) {
  color: var(--vp-c-text-2) !important;
}
</style>
