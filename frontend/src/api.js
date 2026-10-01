// 相对路径：开发时由 Vite 代理；构建后由同一 FastAPI 服务提供网页和 API。
export async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail)
      ? '请输入 1～4000 字的问题，并检查提交格式。'
      : detail || `请求失败（${response.status}）`);
  }
  return data;
}

export function sourceUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : undefined;
  } catch { return undefined; }
}
