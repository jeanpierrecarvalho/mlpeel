import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
  });
}

function jsonResponse(res, status, data) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

/**
 * Start HTTP server to serve the viewer UI and model data.
 * @param {object} modelData - Parsed model data
 * @param {object} options
 * @param {number} options.port
 * @param {string} options.fileName
 * @param {string} [options.modelPath] - absolute path to model file (for inference)
 * @param {string} [options.format] - model format
 * @returns {Promise<{ url: string, close: () => void }>}
 */
export async function startServer(modelData, { port = 8800, fileName = 'model', modelPath = null, format = null } = {}) {
  const clientDir = join(__dirname, 'client');

  const server = createServer((req, res) => {
    const url = new URL(req.url, `http://localhost:${port}`);

    if (url.pathname === '/api/model') {
      jsonResponse(res, 200, { ...modelData, fileName, format, canInfer: format === 'onnx' && !!modelPath });
      return;
    }

    if (url.pathname === '/api/infer' && req.method === 'POST') {
      if (format !== 'onnx' || !modelPath) {
        jsonResponse(res, 400, { error: 'Inference only supported for ONNX models' });
        return;
      }
      handleInfer(req, res, modelPath);
      return;
    }

    // serve static files from client/
    let filePath = url.pathname === '/' ? '/index.html' : url.pathname;
    filePath = join(clientDir, filePath);

    readFile(filePath).then((data) => {
      const ext = extname(filePath);
      res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
      res.end(data);
    }).catch(() => {
      res.writeHead(404);
      res.end('Not found');
    });
  });

  async function handleInfer(req, res, path) {
    try {
      const { runInference } = await import('./inference.js');
      const body = JSON.parse(await readBody(req));
      const result = await runInference(path, body);
      jsonResponse(res, 200, result);
    } catch (err) {
      if (!res.headersSent) jsonResponse(res, 500, { error: err.message });
    }
  }

  const tryListen = (p) => new Promise((resolve, reject) => {
    const onError = (err) => {
      if (err.code === 'EADDRINUSE' && p < port + 10) {
        server.removeListener('error', onError);
        resolve(tryListen(p + 1));
      } else {
        reject(err);
      }
    };
    server.on('error', onError);
    server.listen(p, () => {
      server.removeListener('error', onError);
      resolve({
        url: `http://localhost:${p}`,
        close: () => server.close(),
      });
    });
  });

  return tryListen(port);
}
