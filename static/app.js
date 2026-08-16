const form = document.querySelector('#align-form');
const audioInput = document.querySelector('#audio');
const lyricsInput = document.querySelector('#lyrics');
const dropZone = document.querySelector('#drop-zone');
const emptyState = document.querySelector('#empty-state');
const processingState = document.querySelector('#processing-state');
const errorState = document.querySelector('#error-state');
const results = document.querySelector('#results');
const submitButton = document.querySelector('#submit-button');
const player = document.querySelector('#audio-player');
let currentResult = null;
let timer = null;
let audioUrl = null;

function updateFile(file) {
  if (!file) return;
  document.querySelector('#file-title').textContent = file.name;
  document.querySelector('#file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · ${file.type || '音频文件'}`;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(file);
  player.src = audioUrl;
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
  data.lines.forEach(line => {
    const row = document.createElement('tr');
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
  showState('results');
}

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
  timer = setInterval(() => {
    document.querySelector('#processing-time').textContent = `已用时 ${Math.floor((Date.now() - start) / 1000)} 秒`;
  }, 1000);

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
