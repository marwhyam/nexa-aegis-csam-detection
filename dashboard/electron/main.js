const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 820,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false // <--- CRITICAL FIX: Allows loading local image files
    }
  });

  win.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', ()=> { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

/* IPC handlers */

// Show OS file picker and return selected file paths
ipcMain.handle('dialog:openFiles', async (evt) => {
  const res = await dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Media', extensions: ['jpg','jpeg','png','bmp','tiff','webp','mp4','avi','mov','mkv','flv','wmv'] }]
  });
  return res.canceled ? [] : res.filePaths;
});

// Run Python CLI on provided list of file paths.
ipcMain.handle('python:analyze', async (evt, args) => {
  return new Promise((resolve, reject) => {
    // Determine paths based on where the app is running
    const pythonDir = path.join(__dirname, '..', 'python'); 
    // Fallback: if running from root, check ./python
    const fallbackDir = path.join(__dirname, 'python');
    
    let workingDir = fs.existsSync(pythonDir) ? pythonDir : fallbackDir;
    let csamCli = path.join(workingDir, 'csam_cli.py');

    if (!fs.existsSync(csamCli)) {
        // One last check: maybe we are inside the python folder?
        if (fs.existsSync(path.join(__dirname, 'csam_cli.py'))) {
            workingDir = __dirname;
            csamCli = path.join(__dirname, 'csam_cli.py');
        } else {
            return reject(new Error(`csam_cli.py not found. Searched in: ${pythonDir} and ${fallbackDir}`));
        }
    }

    const outFile = args.out || path.join(workingDir, 'results_from_electron.json');
    const inputs = (args.images || args.paths || []).map(p => path.resolve(p));

    const spawnArgs = [csamCli, '--out', outFile].concat(inputs);
    const py = process.platform === 'win32' ? 'python' : 'python3';

    console.log("Spawning Python:", py, spawnArgs);

    const child = spawn(py, spawnArgs, { cwd: workingDir });

    child.stdout.on('data', (data) => {
      const txt = data.toString();
      evt.sender.send('analysis-log', txt);
    });
    child.stderr.on('data', (data) => {
      const txt = data.toString();
      evt.sender.send('analysis-log', txt);
    });
    child.on('error', (err) => { reject(err); });
    child.on('close', (code) => {
      if (code === 0) {
        try {
          if (fs.existsSync(outFile)) {
            const out = fs.readFileSync(outFile, 'utf8');
            const json = JSON.parse(out);
            resolve({ ok: true, outFile, results: json });
          } else {
            reject(new Error("Python finished but output file missing."));
          }
        } catch (e) {
          reject(e);
        }
      } else {
        reject(new Error(`Python exited with code ${code}`));
      }
    });
  });
});
