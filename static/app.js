const form = document.querySelector('#align-form');
const audioInput = document.querySelector('#audio');
const lyricsInput = document.querySelector('#lyrics');
const dropZone = document.querySelector('#drop-zone');
const emptyState = document.querySelector('#empty-state');
const processingState = document.querySelector('#processing-state');
const errorState = document.querySelector('#error-state');
const results = document.querySelector('#results');
const submitButton = document.querySelector('#submit-button');
const modelSelect = document.querySelector('#model');
const modelBadge = document.querySelector('#model-badge');
const modelStatusText = document.querySelector('#model-status-text');
const modelSize = document.querySelector('#model-size');
const modelProgress = document.querySelector('#model-progress');
const modelProgressBar = document.querySelector('#model-progress-bar');
const modelPathLabel = document.querySelector('#model-path-label');
const modelPath = document.querySelector('#model-path');
const processingStage = document.querySelector('#processing-stage');
const player = document.querySelector('#audio-player');
const timelineWrap = document.querySelector('#timeline-wrap');
const nowLabel = document.querySelector('#lyrics-now-label');
const nowTime = document.querySelector('#lyrics-now-time');
const nowCurrent = document.querySelector('#lyrics-now-current');
const nowNext = document.querySelector('#lyrics-now-next');
let currentResult = null;
let timer = null;
let modelPollTimer = null;
let modelStatuses = new Map();
let audioUrl = null;
let activeLineIndex = -1;

const modelLabels = new Map(
  [...modelSelect.options].map(option => [option.value, option.textContent])
);

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1000)), units.length - 1);
  const value = bytes / (1000 ** index);
  return `${value.toFixed(index >= 3 ? 2 : index === 2 ? 1 : 0)} ${units[index]}`;
}

function statusSuffix(status) {
  if (!status) return '检查中';
  if (status.status === 'downloaded') return '已下载';
  if (status.status === 'downloading') {
    const action = status.phase === 'reconstructing' ? '写入' : '下载';
    return `${action} ${Math.floor(status.progress || 0)}%`;
  }
  if (status.status === 'partial') return '未下载完整';
  if (status.status === 'error') return '下载出错';
  return '未下载';
}

function renderModelOptions() {
  [...modelSelect.options].forEach(option => {
    const suffix = statusSuffix(modelStatuses.get(option.value));
    option.textContent = `${modelLabels.get(option.value)} · ${suffix}`;
  });
}

function renderSelectedModelStatus() {
  const status = modelStatuses.get(modelSelect.value);
  modelBadge.className = 'model-badge';
  if (!status) {
    modelBadge.classList.add('is-checking');
    modelBadge.textContent = '检查中';
    modelStatusText.textContent = '正在读取模型缓存…';
    modelSize.textContent = '';
    modelProgress.hidden = true;
    modelPathLabel.textContent = '缓存位置';
    modelPath.textContent = '—';
    return;
  }

  const progress = Math.max(0, Math.min(100, Number(status.progress) || 0));
  modelBadge.textContent = statusSuffix(status);
  const progressTotalBytes = status.progress_total_bytes || status.total_bytes;
  modelSize.textContent = status.status === 'downloaded'
    ? formatBytes(status.total_bytes)
    : `${formatBytes(status.downloaded_bytes)} / ${formatBytes(progressTotalBytes)}`;
  modelPath.textContent = status.path;

  if (status.status === 'downloaded') {
    modelBadge.classList.add('is-ready');
    modelStatusText.textContent = '模型已下载，可以直接开始';
    modelProgress.hidden = true;
    modelPathLabel.textContent = '已下载到';
  } else if (status.status === 'downloading') {
    modelBadge.classList.add('is-downloading');
    modelStatusText.textContent = status.phase === 'reconstructing'
      ? `正在写入模型文件 · ${progress.toFixed(1)}%`
      : `正在下载网络数据 · ${progress.toFixed(1)}%`;
    modelProgress.hidden = false;
    modelProgressBar.style.width = `${progress}%`;
    modelPathLabel.textContent = '正在下载到';
  } else if (status.status === 'partial') {
    modelBadge.classList.add('is-downloading');
    modelStatusText.textContent = `发现未完成的下载 · ${progress.toFixed(1)}%`;
    modelProgress.hidden = false;
    modelProgressBar.style.width = `${progress}%`;
    modelPathLabel.textContent = '缓存位置';
  } else if (status.status === 'error') {
    modelBadge.classList.add('is-error');
    modelStatusText.textContent = '上次下载没有完成';
    modelProgress.hidden = true;
    modelPathLabel.textContent = '缓存位置';
  } else {
    modelStatusText.textContent = '尚未下载，提交后会自动下载';
    modelProgress.hidden = true;
    modelPathLabel.textContent = '将下载到';
  }
}

function updateProcessingStage() {
  const status = modelStatuses.get(modelSelect.value);
  if (!status) {
    processingStage.textContent = '正在检查识别模型…';
  } else if (status.status === 'downloading') {
    const action = status.phase === 'reconstructing' ? '正在写入模型文件' : '正在下载模型数据';
    const progressTotalBytes = status.progress_total_bytes || status.total_bytes;
    processingStage.textContent = `${action} · ${status.progress.toFixed(1)}% · ${formatBytes(status.downloaded_bytes)} / ${formatBytes(progressTotalBytes)}`;
  } else if (status.status === 'downloaded') {
    processingStage.textContent = '模型已就绪，正在识别并对齐歌词…';
  } else if (status.status === 'error') {
    processingStage.textContent = '模型下载出现问题，正在等待服务返回详情…';
  } else {
    processingStage.textContent = '正在准备识别模型…';
  }
}

async function refreshModelStatuses(updateStage = false) {
  try {
    const response = await fetch('/api/models', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    modelStatuses = new Map(payload.models.map(status => [status.name, status]));
    renderModelOptions();
    renderSelectedModelStatus();
    if (updateStage) updateProcessingStage();
  } catch (error) {
    modelBadge.className = 'model-badge is-error';
    modelBadge.textContent = '状态未知';
    modelStatusText.textContent = '暂时无法读取模型状态';
    modelSize.textContent = '';
  }
}

modelSelect.addEventListener('change', renderSelectedModelStatus);
refreshModelStatuses();

function updateFile(file) {
  if (!file) return;
  document.querySelector('#file-title').textContent = file.name;
  document.querySelector('#file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · ${file.type || '音频文件'}`;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(file);
  player.src = audioUrl;
  updateActiveLyric(true);
}

audioInput.addEventListener('change', () => updateFile(audioInput.files[0]));
lyricsInput.addEventListener('input', () => {
  const count = lyricsInput.value.split(/\r?\n/).filter(line => line.trim()).length;
  document.querySelector('#line-count').textContent = `${count} 行`;
});

['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  audioInput.files = transfer.files;
  updateFile(file);
});

function formatClock(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(2).padStart(5, '0');
  return `${String(minutes).padStart(2, '0')}:${remainder}`;
}

function showState(state) {
  emptyState.hidden = state !== 'empty';
  processingState.hidden = state !== 'processing';
  errorState.hidden = state !== 'error';
  results.hidden = state !== 'results';
}

function renderResult(data) {
  currentResult = data;
  const diagnostics = data.metadata.diagnostics;
  const transcription = data.metadata.transcription || {};
  const quality = Math.round(diagnostics.overall_confidence * 100);
  document.querySelector('#quality-score').textContent = `${quality}%`;
  document.querySelector('#duration').textContent = formatClock(data.metadata.duration);
  document.querySelector('#processing-duration').textContent = `${data.metadata.processing_seconds.toFixed(1)}s`;
  const processingNotes = [];
  if (data.metadata.effective_model && data.metadata.effective_model !== data.metadata.model) {
    processingNotes.push(`${data.metadata.model} 匹配过低，已自动改用 ${data.metadata.effective_model}`);
  }
  if (transcription.prompt_retry) {
    processingNotes.push('首次识别无有效文字，已自动重试');
  }
  const note = processingNotes.length ? ` · ${processingNotes.join('；')}` : '';
  document.querySelector('#result-summary').textContent = `${data.lines.length} 行歌词${note} · 点击任意一行可试听定位`;

  const body = document.querySelector('#timeline-body');
  body.replaceChildren();
  data.lines.forEach((line, lineIndex) => {
    const row = document.createElement('tr');
    row.dataset.lineIndex = lineIndex;
    const values = [line.index, formatClock(line.start), formatClock(line.end), line.text];
    values.forEach((value, index) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      if (index === 1 || index === 2) cell.className = 'time-cell';
      row.appendChild(cell);
    });
    const confidenceCell = document.createElement('td');
    const confidence = document.createElement('span');
    confidence.className = `confidence${line.confidence < .45 ? ' low' : ''}`;
    confidence.textContent = `${Math.round(line.confidence * 100)}%`;
    confidenceCell.appendChild(confidence);
    row.appendChild(confidenceCell);
    row.addEventListener('click', () => {
      player.currentTime = line.start;
      player.play().catch(() => {});
    });
    body.appendChild(row);
  });
  activeLineIndex = -1;
  updateActiveLyric(true);
  showState('results');
}

function lineIndexAtTime(time) {
  const lines = currentResult?.lines || [];
  let low = 0;
  let high = lines.length - 1;
  let candidate = -1;

  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (lines[middle].start <= time) {
      candidate = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }

  if (candidate < 0 || time >= lines[candidate].end) return -1;
  return candidate;
}

function keepActiveRowVisible(row) {
  if (!row || !timelineWrap) return;
  const rowRect = row.getBoundingClientRect();
  const wrapRect = timelineWrap.getBoundingClientRect();
  const headerHeight = timelineWrap.querySelector('thead')?.getBoundingClientRect().height || 0;
  const visibleTop = wrapRect.top + headerHeight;
  if (rowRect.top < visibleTop || rowRect.bottom > wrapRect.bottom) {
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function updateActiveLyric(force = false) {
  const lines = currentResult?.lines || [];
  const nextIndex = lineIndexAtTime(player.currentTime || 0);
  nowTime.textContent = formatClock(player.currentTime || 0);

  if (!force && nextIndex === activeLineIndex) return;

  const previousRow = document.querySelector('#timeline-body tr.is-active');
  previousRow?.classList.remove('is-active');
  previousRow?.removeAttribute('aria-current');
  activeLineIndex = nextIndex;

  if (nextIndex >= 0) {
    const line = lines[nextIndex];
    const row = document.querySelector(`#timeline-body tr[data-line-index="${nextIndex}"]`);
    row?.classList.add('is-active');
    row?.setAttribute('aria-current', 'true');
    nowLabel.textContent = `正在播放 · 第 ${line.index} 行`;
    nowCurrent.textContent = line.text;
    nowNext.textContent = lines[nextIndex + 1] ? `下一句 · ${lines[nextIndex + 1].text}` : '最后一句';
    if (!player.paused) keepActiveRowVisible(row);
    return;
  }

  const upcomingIndex = lines.findIndex(line => line.start > (player.currentTime || 0));
  if (upcomingIndex >= 0) {
    nowLabel.textContent = player.paused ? '已暂停' : '等待下一句';
    nowCurrent.textContent = '♪';
    nowNext.textContent = `下一句 · ${lines[upcomingIndex].text}`;
  } else if (lines.length && player.currentTime >= lines[lines.length - 1].end) {
    nowLabel.textContent = '播放结束';
    nowCurrent.textContent = '歌词播放完毕';
    nowNext.textContent = '';
  } else {
    nowLabel.textContent = '等待播放';
    nowCurrent.textContent = '点击播放，歌词会跟随音频高亮';
    nowNext.textContent = lines[0] ? `第一句 · ${lines[0].text}` : '';
  }
}

player.addEventListener('timeupdate', () => updateActiveLyric());
player.addEventListener('seeking', () => updateActiveLyric(true));
player.addEventListener('play', () => updateActiveLyric(true));
player.addEventListener('pause', () => {
  updateActiveLyric(true);
  if (activeLineIndex >= 0) nowLabel.textContent = `已暂停 · 第 ${currentResult.lines[activeLineIndex].index} 行`;
});
player.addEventListener('ended', () => updateActiveLyric(true));

function parseError(payload, status) {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (Array.isArray(payload?.detail)) return payload.detail.map(item => item.msg).join('；');
  return `请求失败（HTTP ${status}）`;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!audioInput.files[0]) return;
  showState('processing');
  submitButton.disabled = true;
  currentResult = null;
  const start = Date.now();
  clearInterval(timer);
  clearInterval(modelPollTimer);
  updateProcessingStage();
  refreshModelStatuses(true);
  timer = setInterval(() => {
    document.querySelector('#processing-time').textContent = `已用时 ${Math.floor((Date.now() - start) / 1000)} 秒`;
  }, 1000);
  modelPollTimer = setInterval(() => refreshModelStatuses(true), 750);

  try {
    const response = await fetch('/api/align', { method: 'POST', body: new FormData(form) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(parseError(payload, response.status));
    renderResult(payload);
  } catch (error) {
    document.querySelector('#error-message').textContent = error.message || String(error);
    showState('error');
  } finally {
    clearInterval(timer);
    clearInterval(modelPollTimer);
    refreshModelStatuses();
    submitButton.disabled = false;
  }
});

document.querySelectorAll('[data-format]').forEach(button => button.addEventListener('click', () => {
  if (!currentResult) return;
  const format = button.dataset.format;
  const content = currentResult.exports[format];
  const mimeTypes = { json: 'application/json', csv: 'text/csv', lrc: 'text/plain', srt: 'text/plain', jsx: 'text/plain' };
  const blob = new Blob([content], { type: `${mimeTypes[format]};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const originalName = currentResult.metadata.filename.replace(/\.[^.]+$/, '');
  link.href = url;
  link.download = `${originalName}.${format}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
