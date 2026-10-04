/**
 * script.js
 * Controls frontend interactivity, drag-and-drop, API requests,
 * live telemetry stats rendering, theme switching, and exports.
 */

const $ = (id) => document.getElementById(id);

// DOM Elements
const fileInput = $('fileInput');
const dropZone = $('dropZone');
const dropPrompt = $('dropPrompt');
const previewContainer = $('previewContainer');
const imagePreview = $('imagePreview');
const removeImgBtn = $('removeImgBtn');

const processBtn = $('processBtn');
const processBtnText = $('processBtnText');
const clearBtn = $('clearBtn');
const cvPreprocessing = $('cvPreprocessing');

const output = $('output');
const statStatus = $('statStatus');
const statConfidence = $('statConfidence');
const statWords = $('statWords');
const statLatency = $('statLatency');
const statusIndicator = $('statusIndicator');

const copyBtn = $('copyBtn');
const downloadBtn = $('downloadBtn');
const exportJsonBtn = $('exportJsonBtn');

const themeBtn = $('themeBtn');
const themeIcon = $('themeIcon');
const themeLabel = $('themeLabel');

let selectedFile = null;
let lastResultData = null;

/* ---------- Theme Handling (Black / White) ---------- */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);

  if (theme === 'dark') {
    if (themeIcon) themeIcon.textContent = '☀️';
    if (themeLabel) themeLabel.textContent = 'THEME: TRUE VOID';
  } else {
    if (themeIcon) themeIcon.textContent = '🌙';
    if (themeLabel) themeLabel.textContent = 'THEME: PURE LIGHT';
  }
}

// Initial theme load
const initialTheme = document.documentElement.getAttribute('data-theme') || 'dark';
applyTheme(initialTheme);

themeBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

/* ---------- Status Feedback Stream ---------- */
let statusTimer = null;
function streamStatus(msg) {
  statusIndicator.textContent = msg;
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => {
    statusIndicator.textContent = selectedFile ? 'IMAGE READY' : 'STANDBY';
  }, 3500);
}

/* ---------- File Selection & Handling ---------- */
function setFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    streamStatus('ERR: INVALID IMAGE');
    return;
  }

  selectedFile = file;
  imagePreview.src = URL.createObjectURL(file);
  previewContainer.hidden = false;
  dropPrompt.hidden = true;

  processBtn.disabled = false;
  clearBtn.disabled = false;

  statStatus.textContent = 'IMAGE ATTACHED';
  streamStatus('IMAGE LOADED');
}

function clearAll() {
  selectedFile = null;
  lastResultData = null;
  fileInput.value = '';

  previewContainer.hidden = true;
  imagePreview.removeAttribute('src');
  dropPrompt.hidden = false;

  processBtn.disabled = true;
  clearBtn.disabled = true;
  copyBtn.disabled = true;
  downloadBtn.disabled = true;
  if (exportJsonBtn) exportJsonBtn.disabled = true;

  output.value = '';
  statStatus.textContent = 'STANDBY';
  statConfidence.textContent = '--%';
  statWords.textContent = '0';
  statLatency.textContent = '0ms';

  streamStatus('BUFFER CLEARED');
}

// Dropzone interactions
dropZone.addEventListener('click', (e) => {
  if (e.target !== removeImgBtn && !removeImgBtn.contains(e.target)) {
    fileInput.click();
  }
});

dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    setFile(fileInput.files[0]);
  }
});

removeImgBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearAll();
});

// Drag & drop events
['dragenter', 'dragover'].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add('over');
  });
});

['dragleave', 'drop'].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove('over');
  });
});

dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
    setFile(e.dataTransfer.files[0]);
  }
});

// Clipboard paste listener (Cmd/Ctrl + V)
window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;

  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const blob = item.getAsFile();
      setFile(blob);
      streamStatus('PASTED FROM CLIPBOARD');
      return;
    }
  }
});

// Sample buttons
document.querySelectorAll('[data-sample]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const sampleUrl = btn.dataset.sample;
    streamStatus('FETCHING SAMPLE...');
    try {
      const res = await fetch(sampleUrl);
      const blob = await res.blob();
      const fileName = sampleUrl.split('/').pop() || 'sample.png';
      setFile(new File([blob], fileName, { type: blob.type || 'image/png' }));
      // Run OCR immediately
      setTimeout(() => runOCR(), 150);
    } catch (err) {
      streamStatus('ERR: SAMPLE FETCH FAILED');
    }
  });
});

clearBtn.addEventListener('click', clearAll);
processBtn.addEventListener('click', runOCR);

/* ---------- OCR Pipeline Execution ---------- */
async function runOCR() {
  if (!selectedFile) return;

  processBtn.disabled = true;
  processBtnText.textContent = 'DECODING TENSORS...';
  statStatus.textContent = 'PROCESSING...';
  streamStatus('RUNNING NEURAL OCR');

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('enable_preprocessing', cvPreprocessing.checked);
  form.append('spell_check', false);
  form.append('psm', 3);

  try {
    const res = await fetch('/api/read-image', { method: 'POST', body: form });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();

    if (data.status !== 'success') {
      throw new Error(data.message || 'Extraction failed');
    }

    lastResultData = data;
    output.value = data.proper_text || '';

    // Update Telemetry
    statStatus.textContent = data.variant ? data.variant.toUpperCase() : 'DECODED';
    statConfidence.textContent = `${data.confidence}%`;
    statWords.textContent = data.word_count;
    statLatency.textContent = `${data.inference_time_ms}ms`;

    const hasText = Boolean(data.proper_text && data.proper_text.trim());
    copyBtn.disabled = !hasText;
    downloadBtn.disabled = !hasText;
    if (exportJsonBtn) exportJsonBtn.disabled = !hasText;

    streamStatus(hasText ? 'DECODING COMPLETE' : 'NO GLYPHS FOUND');
  } catch (err) {
    // If backend API is not found (e.g. static hosting on GitHub Pages),
    // run the offline ML model directly in-browser using WebAssembly!
    if (window.Tesseract) {
      streamStatus('RUNNING IN-BROWSER WASM OCR');
      statStatus.textContent = 'CLIENT WASM ENGINE';
      const t0 = performance.now();

      try {
        const workerResult = await Tesseract.recognize(selectedFile, 'eng', {
          logger: (m) => {
            if (m.status === 'recognizing text' && m.progress) {
              statStatus.textContent = `WASM: ${(m.progress * 100).toFixed(0)}%`;
            }
          }
        });

        const t1 = performance.now();
        const extractedText = (workerResult?.data?.text || '').trim();
        const confidence = Math.round(workerResult?.data?.confidence || 0);

        output.value = extractedText;
        lastResultData = {
          proper_text: extractedText,
          raw_text: extractedText,
          confidence: confidence,
          word_count: extractedText.split(/\s+/).filter(Boolean).length,
          character_count: extractedText.length,
          inference_time_ms: Math.round(t1 - t0),
          variant: 'wasm-neural'
        };

        statStatus.textContent = 'WASM DECODED';
        statConfidence.textContent = `${confidence}%`;
        statWords.textContent = lastResultData.word_count;
        statLatency.textContent = `${lastResultData.inference_time_ms}ms`;

        const hasText = Boolean(extractedText);
        copyBtn.disabled = !hasText;
        downloadBtn.disabled = !hasText;
        if (exportJsonBtn) exportJsonBtn.disabled = !hasText;

        streamStatus(hasText ? 'WASM DECODING COMPLETE' : 'NO GLYPHS FOUND');
        return;
      } catch (wasmErr) {
        statStatus.textContent = 'FAILED';
        streamStatus(`ERR: ${wasmErr.message}`);
        return;
      }
    }

    statStatus.textContent = 'FAILED';
    streamStatus(`ERR: ${err.message}`);
  } finally {
    processBtn.disabled = false;
    processBtnText.textContent = 'EXTRACT & DECODE TEXT';
  }
}

/* ---------- Output Action Handlers ---------- */
copyBtn.addEventListener('click', async () => {
  if (!output.value) return;

  try {
    await navigator.clipboard.writeText(output.value);
    streamStatus('BUFFER COPIED');
  } catch (_) {
    output.select();
    document.execCommand('copy');
    streamStatus('BUFFER COPIED');
  }
});

downloadBtn.addEventListener('click', () => {
  if (!output.value) return;

  const blob = new Blob([output.value], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `textflow_export_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  streamStatus('EXPORTED .TXT FILE');
});

if (exportJsonBtn) {
  exportJsonBtn.addEventListener('click', () => {
    if (!lastResultData) return;

    const payload = {
      decoded_text: output.value,
      raw_ocr: lastResultData.raw_text,
      confidence_percent: lastResultData.confidence,
      word_count: lastResultData.word_count,
      character_count: lastResultData.character_count,
      latency_ms: lastResultData.inference_time_ms,
      variant: lastResultData.variant,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `textflow_data_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    streamStatus('EXPORTED JSON DATA');
  });
}
