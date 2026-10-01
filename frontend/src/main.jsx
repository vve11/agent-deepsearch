import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { request, sourceUrl } from './api';
import './styles.css';

const phaseLabels = { planning: '正在规划', planned: '准备研究', research: '正在研究' };
const eventLabels = {
  run_started: '创建研究任务', run_resumed: '继续研究', run_completed: '研究完成', run_failed: '研究中断',
  plan_started: '拆分研究问题', plan_generated: '生成研究计划', plan_saved: '计划已保存', plan_reused: '复用研究计划',
  plan_failed: '规划未完成', model_started: '请求模型', model_completed: '模型已响应', model_failed: '模型请求失败',
  model_retrying: '等待重试', tool_started: '搜索资料', tool_completed: '搜索已返回', tool_failed: '搜索失败',
  tool_skipped: '工具预算已用尽', checkpoint_saved: '保存恢复点', citation_check_passed: '引用编号检查通过',
  citation_check_failed: '发现未知引用', repair_started: '修复引用', repair_completed: '引用修复完成', repair_failed: '引用修复失败',
};
const examples = ['比较电动汽车电池回收的主要技术路线，附上来源。', '研究钠离子电池的应用场景与现阶段的局限。', '梳理光伏组件回收的主要方法和经济性。'];
function statusLabel(run) {
  if (run.active) return phaseLabels[run.phase] || '运行中';
  if (run.status === 'completed') return '已完成';
  if (run.status === 'failed') return '已中断';
  return '待继续';
}
function Status({ run }) {
  return <span className={`status ${run.active ? 'live' : run.status}`}><i />{statusLabel(run)}</span>;
}
function initialSelection() {
  const id = window.location.hash.slice(1);
  return /^[a-f0-9]{32}$/.test(id) ? id : null;
}
function App() {
  const [selected, setSelected] = useState(initialSelection);
  const [runs, setRuns] = useState([]);
  const [detail, setDetail] = useState(null);
  const [health, setHealth] = useState(null);
  const [question, setQuestion] = useState('');
  const [tab, setTab] = useState('answer');
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [sending, setSending] = useState(false);
  const [limit, setLimit] = useState(30);
  const [refreshKey, setRefreshKey] = useState(0);
  const textarea = useRef(null);

  useEffect(() => {
    const onHash = () => setSelected(initialSelection());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // 完成一次刷新后再等待两秒，避免慢请求重叠。切换任务时中止旧请求。
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    setDetail(null);
    async function poll() {
      try {
        const options = { signal: controller.signal };
        const [nextHealth, list] = await Promise.all([
          request('/health', options), request(`/runs?limit=${limit}`, options),
        ]);
        if (controller.signal.aborted) return;
        setHealth(nextHealth); setRuns(list.items);
        if (selected) {
          const record = await request(`/runs/${selected}`, options);
          if (!controller.signal.aborted) setDetail(record);
        }
        if (!controller.signal.aborted) setPollError('');
      } catch (e) {
        if (!controller.signal.aborted) setPollError(e.message || '无法连接本地服务');
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selected, limit, refreshKey]);

  function select(id) {
    window.history.replaceState(null, '', id ? `#${id}` : window.location.pathname);
    setSelected(id); setTab('answer'); setError('');
    if (!id) setTimeout(() => textarea.current?.focus(), 0);
  }
  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || sending) return;
    setSending(true); setError('');
    try {
      const result = await request('/runs', { method: 'POST', body: JSON.stringify({ question }) });
      select(result.run_id); setQuestion(''); setRefreshKey(v => v + 1);
    } catch (e) { setError(e.message); }
    finally { setSending(false); }
  }
  async function resume() {
    setSending(true); setError('');
    try {
      await request(`/runs/${selected}/resume`, { method: 'POST' });
      setRefreshKey(v => v + 1);
    } catch (e) { setError(e.message); }
    finally { setSending(false); }
  }
  const busy = health?.busy;
  const canSubmit = health?.model_configured && !busy && !sending && question.trim() && !pollError;

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); select(null); }}>
        <span className="brand-mark">研</span><span>研途<small>RESEARCH WORKSPACE</small></span>
      </a>
      <button className="new-button" onClick={() => select(null)}><span>＋</span> 新建研究</button>
      <div className="sidebar-label">研究记录 <span>{runs.length}</span></div>
      <nav className="history" aria-label="研究记录">
        {runs.length === 0 && <p className="empty-history">你的研究将保存在这里。<br />刷新网页也不会丢失。</p>}
        {runs.map(run => <button key={run.id} className={`history-item ${selected === run.id ? 'selected' : ''}`} onClick={() => select(run.id)}>
          <span className="history-title">{run.question}</span>
          <Status run={run} />
        </button>)}
        {runs.length >= limit && limit < 100 && <button className="text-button" onClick={() => setLimit(Math.min(limit + 30, 100))}>显示更多记录</button>}
        {runs.length === 100 && <p className="empty-history">显示最近 100 个任务</p>}
      </nav>
      <div className="sidebar-bottom"><span className={`connection-dot ${pollError ? 'offline' : ''}`} />{pollError ? '连接暂时中断' : health ? '本地工作空间' : '正在连接'}<small>记录保存在你的电脑上</small></div>
    </aside>

    <main>
      <header className="topbar"><span>研究工作台 <span className="breadcrumb">/ {selected ? '任务详情' : '新研究'}</span></span><span className="local-tag">LOCAL <span>·</span> 单任务运行</span></header>
      <div className="main-content">
        {(error || pollError) && <div role="alert" className="notice error">{error || pollError}<button onClick={() => { setError(''); setRefreshKey(v => v + 1); }}>重新加载</button></div>}
        {health && !health.model_configured && <div className="notice">模型尚未配置。请在项目 .env 中设置 DEEPSEEK_API_KEY 并重启服务。历史记录仍可查看。</div>}

        {!selected ? <section className="welcome">
          <div className="eyebrow"><span /> 从一个好问题开始</div>
          <h1>让问题，走向有据可循的答案。</h1>
          <p className="intro">提出你想探索的问题。研究助手会制定计划、搜索资料，<br className="desktop-break" />并将结论与来源一起交给你。</p>
          <form className="composer" onSubmit={submit}>
            <label className="sr-only" htmlFor="question">研究问题</label>
            <textarea id="question" ref={textarea} value={question} onChange={e => setQuestion(e.target.value)} maxLength={4000} placeholder="你想研究什么？可以说明关注的角度、比较对象或时间范围……" rows={5} />
            <div className="composer-footer"><span>{busy ? '已有研究运行中，可先准备下一个问题' : '自动规划 · 网页搜索 · 来源引用'}</span><button className="primary" disabled={!canSubmit}>{sending ? '正在提交…' : '开始研究'} <span>↗</span></button></div>
          </form>
          <div className="suggestions"><span>试试这些问题</span>{examples.map((example, index) => <button key={example} onClick={() => { setQuestion(example); textarea.current?.focus(); }}><b>0{index + 1}</b>{example}<span>↗</span></button>)}</div>
          <div className="process-strip"><div><b>01</b><span>制定计划<small>拆分值得调查的问题</small></span></div><div><b>02</b><span>寻找证据<small>搜索并登记资料来源</small></span></div><div><b>03</b><span>形成回答<small>保留引用与完整过程</small></span></div></div>
        </section> : !detail ? <div className="loading-state">{pollError ? '暂时无法读取此任务，请稍后重试。' : '正在读取研究记录…'}</div> : <section className="research-detail">
          <div className="detail-meta"><span className="eyebrow">RESEARCH NOTES</span><Status run={detail} /></div>
          <h1>{detail.question}</h1>
          <div className="detail-actions"><span className="run-id" title={detail.id}>任务 {detail.id.slice(0, 12)}</span>{detail.can_resume && <button className="primary" disabled={busy || sending || !health?.model_configured} onClick={resume}>{sending ? '正在恢复…' : '继续研究'} ↗</button>}</div>
          {detail.active && <div className="running-note" role="status"><span className="pulse" />研究正在后台进行，页面每两秒刷新。你可以查看其他记录，或稍后回来。</div>}
          {!detail.active && detail.status === 'running' && <div className="notice">当前没有后台任务在执行。研究可能因服务重启而中断；如恢复状态有效，可以点击“继续研究”。</div>}
          {detail.error && <div className="notice error"><strong>研究已中断</strong><p>{detail.error}</p>{detail.can_resume && <small>已保存的计划或检查点可用于继续。</small>}</div>}
          {detail.warning && <div className="notice">{detail.warning}</div>}
          {detail.plan && <div className="plan-card"><div className="section-label">研究计划 <span>{detail.plan.tasks.length} 个调查方向</span></div>{detail.plan.tasks.map(task => <div className="plan-task" key={task.id}><span>{task.id}</span><p>{task.question}</p></div>)}</div>}
          <div className="tabs" role="tablist" aria-label="任务内容">{[['answer', '研究回答'], ['sources', `资料来源 ${detail.evidence.length}`], ['events', '执行过程']].map(([key, label]) => <button key={key} role="tab" aria-selected={tab === key} onClick={() => setTab(key)} className={tab === key ? 'active' : ''}>{label}</button>)}</div>
          <div role="tabpanel">
            {tab === 'answer' && (detail.answer ? <article className="answer"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{a: ({children, href}) => <a href={sourceUrl(href)} target="_blank" rel="noopener noreferrer">{children}</a>}}>{detail.answer}</ReactMarkdown><div className="answer-footnote">引用编号检查保证来源编号存在，不代表所有结论都已通过事实核验。</div></article> : <div className="empty-panel"><div className="empty-symbol">◎</div><h3>{detail.active ? '让资料汇成答案' : '还没有最终回答'}</h3><p>{detail.active ? '你可以先查看研究计划和执行过程。' : '继续研究后，完成的回答会显示在这里。'}</p></div>)}
            {tab === 'sources' && (detail.evidence.length ? <div className="sources">{detail.evidence.map(item => <article className="source-card" key={item.id}><span className="source-id">{item.id}</span><div><h3><a href={sourceUrl(item.url)} target="_blank" rel="noopener noreferrer">{item.title || '未命名来源'} ↗</a></h3><p className="source-url">{item.url}</p><details><summary>查看资料摘要</summary><p className="source-content">{item.content}</p></details></div></article>)}</div> : <div className="empty-panel"><h3>暂未登记来源</h3><p>搜索完成并保存后，资料会出现在这里。</p></div>)}
            {tab === 'events' && <div className="timeline"><p className="muted">最近 {detail.events.length} 条事件 · 完整记录保存在数据库</p>{detail.events.map(event => <div className="event" key={event.id}><i /><div><div className="event-heading"><strong>{eventLabels[event.event_type] || event.event_type}</strong><time>{new Date(event.created_at).toLocaleTimeString('zh-CN', {hour12: false})}</time></div><details><summary>{event.event_type}</summary><pre>{JSON.stringify(event.payload, null, 2)}</pre></details></div></div>)}</div>}
          </div>
        </section>}
      </div>
      <footer className="page-footer">研途 · 把探索过程留下来 <span>规划 / 证据 / 可恢复研究</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
