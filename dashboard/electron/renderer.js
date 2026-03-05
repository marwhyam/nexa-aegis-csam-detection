// --- DOM ELEMENTS ---
const btnUpload = document.getElementById('btn-upload');
const dropArea = document.getElementById('dropArea');
const processBtn = document.getElementById('processBtn');
const tableBody = document.getElementById('rows');
const themeToggle = document.getElementById('themeToggle');

// Stats Counters
const statTotal = document.getElementById('statTotal');
const statCSAM = document.getElementById('statCSAM');
const statAdult = document.getElementById('statAdult');
const statMinorsNSFW = document.getElementById('statMinorsNSFW');

// Modal Elements
const modal = document.getElementById('evidenceModal');
const modalImg = document.getElementById('modalImagePlaceholder');
const closeModalBtn = document.getElementById('closeModalBtn');
const revealBtn = document.getElementById('revealBtn');
const modalRiskLabel = document.getElementById('modalRiskLabel');

// Global Variables
let filesToProcess = [];
let currentAnalysisResults = [];
let riskChart = null;

function updateRiskChart(csamCount, adultCount) {
  const canvas = document.getElementById('riskChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (!riskChart) {
    riskChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['CSAM', 'Adult / NSFW'],
        datasets: [{
          label: 'Cases',
          data: [csamCount, adultCount],
          backgroundColor: [
            'rgba(239, 68, 68, 0.8)',   // red
            'rgba(245, 158, 11, 0.8)'   // amber
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.y} cases`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#9CA3AF' },
            grid: { display: false }
          },
          y: {
            beginAtZero: true,
            ticks: { color: '#6B7280', precision: 0 },
            grid: { color: 'rgba(55,65,81,0.4)' }
          }
        }
      }
    });
  } else {
    riskChart.data.datasets[0].data = [csamCount, adultCount];
    riskChart.update();
  }
}


// --- 1. LIGHT/DARK MODE TOGGLE ---
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    themeToggle.innerHTML = isLight
        ? '<span class="material-icons-round">dark_mode</span> Dark Mode'
        : '<span class="material-icons-round">light_mode</span> Light Mode';
});

// --- 2. HANDLE FILE UPLOADS ---
btnUpload.addEventListener('click', async () => {
    const paths = await window.api.openFilePicker();
    if (paths && paths.length > 0) addFilesToQueue(paths);
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        const paths = Array.from(e.dataTransfer.files)
            .map(f => f.path)
            .filter(p => p);
        addFilesToQueue(paths);
    }
});

function addFilesToQueue(paths) {
    const newPaths = paths.filter(p => !filesToProcess.includes(p));
    filesToProcess = [...filesToProcess, ...newPaths];
    const label = dropArea.querySelector('.small');
    if (label)
        label.innerHTML = `<span style="color:var(--cyan-neon); font-weight:700;">${filesToProcess.length} files ready</span> for analysis.`;
}

// --- 3. RUN ANALYSIS PROCESS ---
processBtn.addEventListener('click', async () => {
    if (filesToProcess.length === 0) return alert('Please upload files first.');

    processBtn.disabled = true;
    processBtn.innerHTML =
        '<span class="material-icons-round" style="animation:spin 1s linear infinite">refresh</span> PROCESSING...';

    let counts = { total: 0, csam: 0, adult: 0, minorsNsfw: 0 };
    tableBody.innerHTML = '';
    currentAnalysisResults = [];

    try {
        const response = await window.api.analyzeFiles({ paths: filesToProcess });

        if (response.results) {
            currentAnalysisResults = response.results;

            currentAnalysisResults.forEach((item, index) => {
                counts.total++;

                const mediaType = item.media_type || (item.video_path ? 'video' : 'image');
                let rawPath = item.image_path || item.video_path || item.file_path || item.filename || 'Unknown';
                let displayName = rawPath.split(/[\\/]/).pop();
                let displayFull = `<div style="font-family: monospace; color: var(--text-muted); font-size: 0.7rem;">${rawPath}</div>`;

                let isCSAM = item.csam_detected || (item.csam_frames && item.csam_frames > 0) || false;
                let isNSFW = item.nsfw_detected || (item.nsfw_frames && item.nsfw_frames > 0) || false;
                let faceCount = item.faces_detected || item.total_faces || 0;
                let under18Count = item.under18_faces || 0;
                let nsfwConf = item.nsfw_confidence != null
                  ? Math.round(item.nsfw_confidence * 100)
                  : 0;
                if (mediaType === 'video') {
                  nsfwConf = item.nsfw_frames ? 100 : 0;
                  under18Count = item.under18_frames || 0;
                }


                let priorityClass = 'p-low';
                let priorityLabel = 'CLEAN';
                let detectColor = 'var(--cyan-neon)';
                let detectText = 'Safe Content';

                if (isCSAM) {
                    counts.csam++;
                    counts.minorsNsfw++;
                    priorityClass = 'p-high';
                    priorityLabel = 'HIGH RISK';
                    detectColor = 'var(--red-neon)';
                    detectText = 'CSAM Detected';
                } else if (isNSFW) {
                    counts.adult++;
                    priorityClass = 'p-med';
                    priorityLabel = 'MEDIUM';
                    detectColor = 'var(--amber-neon)';
                    detectText = 'Adult / NSFW';
                }

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-family: monospace; color: var(--text-muted); font-size: 0.85rem;">
                     ${displayName}
                     ${displayFull}
                    </td>
                    <td><span class="priority-badge ${priorityClass}">${priorityLabel}</span></td>
                    <td style="color: ${detectColor}; font-weight:600;">${detectText}${mediaType === 'video' ? ' (Video)' : ''}</td>
                    <td>
                        ${mediaType === 'video'
                          ? `Frames: ${item.processed_frames || 0}/${item.total_frames || 0}<br>CSAM frames: ${item.csam_frames || 0} | NSFW frames: ${item.nsfw_frames || 0}`
                          : `${faceCount} Faces ${under18Count > 0 ? `<span style="color:var(--purple-neon); font-weight:700;">(${under18Count} Minor)</span>` : ''}`
                        }
                    </td>
                    <td>${nsfwConf}%</td>
                    <td>
                        <button class="btn-view" data-index="${index}" ${mediaType === 'video' ? 'disabled title="Preview not available for videos"' : ''}>
                            <span class="material-icons-round" style="font-size:18px;">visibility</span> View
                        </button>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        }

        statTotal.innerText = counts.total;
        statCSAM.innerText = counts.csam;
        statAdult.innerText = counts.adult;
        statMinorsNSFW.innerText = counts.minorsNsfw;

        updateRiskChart(counts.csam, counts.adult);

        if (counts.csam > 0) {
            processBtn.style.background = 'var(--red-neon)';
            processBtn.innerText = '⚠️ THREATS DETECTED';
        } else {
            processBtn.innerText = '✔ ANALYSIS COMPLETE';
            setTimeout(() => {
                processBtn.innerHTML = '▶ BEGIN ANALYSIS';
                processBtn.style.background = 'var(--cyan-neon)';
                processBtn.disabled = false;
            }, 3000);
        }
        addSessionToRecent(`Case_${new Date().toISOString().split('T')[0]}`, currentAnalysisResults);

    } catch (err) {
        console.error(err);
        alert('Pipeline Error: ' + err.message);
        processBtn.disabled = false;
        processBtn.innerText = '▶ BEGIN ANALYSIS';
    }
});

// --- 4. MODAL LOGIC (Integrated Here) ---
window.openEvidenceModal = function (imagePath, riskLevel) {
    if (!imagePath) return alert('Error: No image path provided.');

    let displayPath = imagePath.replace(/\\/g, '/');
    if (!displayPath.startsWith('file://')) {
        if (!displayPath.startsWith('/')) {
            displayPath = '/' + displayPath;
        }
        displayPath = 'file://' + displayPath;
    }

    console.log('Opening Modal for:', displayPath);

    modalImg.src = displayPath;
    if (modalRiskLabel) modalRiskLabel.innerText = riskLevel + ' CONTENT';

    modalImg.classList.remove('revealed');
    if (revealBtn) {
        revealBtn.innerText = 'REVEAL IMAGE';
        revealBtn.style.background = 'var(--red-neon)';
        revealBtn.style.color = 'white';
    }

    modal.classList.add('active');
};

// Close Button
if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => {
        modal.classList.remove('active');
        modalImg.src = '';
    });
}

// Reveal/Obscure Button
if (revealBtn) {
    revealBtn.addEventListener('click', () => {
        if (modalImg.classList.contains('revealed')) {
            modalImg.classList.remove('revealed');
            revealBtn.innerText = 'REVEAL IMAGE';
            revealBtn.style.background = 'var(--red-neon)';
            revealBtn.style.color = 'white';
        } else {
            modalImg.classList.add('revealed');
            revealBtn.innerText = 'OBSCURE IMAGE';
            revealBtn.style.background = 'var(--bg-card)';
            revealBtn.style.color = 'var(--text-muted)';
        }
    });
}

// --- 5. VIEW BUTTON CLICK HANDLER ---
tableBody.addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-view');
    if (!btn) return;

    const index = btn.dataset.index;
    const item = currentAnalysisResults[index];

    if (!item) return alert('Error: Data missing for this item.');

    let rawPath = item.image_path || item.file_path || item.filename || '';
    if (item.media_type === 'video' || item.video_path) {
        alert('Preview not available for videos.');
        return;
    }
    let blurredPath = item.blurred_image_path || null;
    let label = 'CLEAN';
    if (item.csam_detected) label = 'HIGH RISK';
    else if (item.nsfw_detected) label = 'MEDIUM';

    if (window.openEvidenceModal) {
        const displayPath = blurredPath || rawPath;
        window.openEvidenceModal(displayPath, label);
    } else {
        console.error('openEvidenceModal function missing on window');
        alert('Interface Error: Modal function not initialized.');
    }
});
const recentSection = document.querySelector('.recent-section');

function addSessionToRecent(sessionName, results) {
  const div = document.createElement('div');
  div.classList.add('recent-item');
  div.innerHTML = `
    <span class="material-icons-round" style="font-size:16px; color:var(--cyan-neon)">folder</span>
    ${sessionName}
    <button class="btn-view" style="margin-left: 10px; font-size: 0.7rem; padding: 4px 8px;" data-session="${sessionName}">
      <span class="material-icons-round" style="font-size:14px;">description</span> Report
    </button>
  `;
  recentSection.appendChild(div);

  div.querySelector('button').addEventListener('click', () => {
    generateSessionReport(sessionName, results);
  });
}

function generateSessionReport(sessionName, results) {
  let html = `
  <html><head><title>${sessionName} - Nexa Aegis Report</title></head><body style="font-family: Inter, sans-serif;">
  <h2>${sessionName}</h2>
  <p><b>Total Processed:</b> ${results.length}</p>
  <table border="1" cellpadding="6" cellspacing="0">
  <tr><th>File</th><th>Priority</th><th>Detection</th></tr>
  ${results.map(r => `
    <tr>
      <td>${r.image_path || r.filename}</td>
      <td>${r.csam_detected ? 'HIGH RISK' : r.nsfw_detected ? 'MEDIUM' : 'CLEAN'}</td>
      <td>${r.csam_detected ? 'CSAM' : r.nsfw_detected ? 'NSFW' : 'Safe'}</td>
    </tr>
  `).join('')}
  </table>
  </body></html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  window.open(url, '_blank');
}

// --- SPIN ANIMATION ---
const styleSheet = document.createElement('style');
styleSheet.innerText = `
@keyframes spin { 100% { transform: rotate(360deg); } }
`;
document.head.appendChild(styleSheet);



