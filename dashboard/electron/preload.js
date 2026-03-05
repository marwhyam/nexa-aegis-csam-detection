const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  openFilePicker: () => ipcRenderer.invoke('dialog:openFiles'),
  analyzeFiles: (args) => ipcRenderer.invoke('python:analyze', args),
  onLog: (cb) => ipcRenderer.on('analysis-log', (evt, data) => cb(data))
});
