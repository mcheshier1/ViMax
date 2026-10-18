const PROVIDER_DOMAINS = [
  ['openrouter.ai', 'openrouter'],
  ['agnes-ai.com', 'agnes'],
  ['yunwu.ai', 'yunwu'],
  ['ltx.io', 'ltx'],
  ['openai.com', 'openai'],
  ['anthropic.com', 'anthropic'],
  ['cohere.com', 'cohere'],
  ['cohere.ai', 'cohere'],
  ['jina.ai', 'jina'],
];

function rawPathname(value) {
  const authorityStart = value.indexOf('//') + 2;
  const suffixStart = value.slice(authorityStart).search(/[/?#]/);
  if (suffixStart < 0 || value[authorityStart + suffixStart] !== '/') return '';
  const suffix = value.slice(authorityStart + suffixStart);
  const pathEnd = suffix.search(/[?#]/);
  const pathname = pathEnd < 0 ? suffix : suffix.slice(0, pathEnd);
  return pathname.replace(/\/+$/, '');
}

export function credentialProvider(section = {}) {
  const baseUrl = typeof section?.base_url === 'string' ? section.base_url.trim() : '';
  if (baseUrl) {
    try {
      const url = new URL(baseUrl);
      if (url.protocol === 'http:' || url.protocol === 'https:') {
        const hostname = url.hostname.toLowerCase();
        for (const [domain, provider] of PROVIDER_DOMAINS) {
          if (hostname === domain || hostname.endsWith(`.${domain}`)) return provider;
        }
        const pathname = rawPathname(baseUrl);
        return `endpoint:${url.origin}${pathname}`;
      }
    } catch {
      // Invalid or non-HTTP URLs use the configured provider label below.
    }
  }

  const modelProvider = typeof section?.model_provider === 'string' ? section.model_provider.trim().toLowerCase() : '';
  const provider = typeof section?.provider === 'string' ? section.provider.trim().toLowerCase() : '';
  return `provider:${modelProvider || provider || 'default'}`;
}
