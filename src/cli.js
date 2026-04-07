#!/usr/bin/env node

import { resolve, basename } from 'node:path';
import { existsSync } from 'node:fs';
import { exec } from 'node:child_process';
import { detect } from './parsers/detect.js';
import { parse as parseOnnx } from './parsers/onnx.js';
import { parse as parseSafetensors } from './parsers/safetensors.js';
import { parse as parseGguf } from './parsers/gguf.js';
import { startServer } from './server.js';

const PARSERS = {
  onnx: parseOnnx,
  safetensors: parseSafetensors,
  gguf: parseGguf,
};

function printHelp() {
  console.log(`
  mlpeel — zero-dependency neural network model viewer

  Usage:
    mlpeel <model-file> [options]
    npx mlpeel model.onnx

  Options:
    --port <n>       Server port (default: 8800)
    --no-open        Don't open browser automatically
    --json           Output model info as JSON (no server)
    -h, --help       Show this help
    -v, --version    Show version

  Supported formats:
    .onnx            ONNX models
    .safetensors     Safetensors (HuggingFace)
    .gguf            GGUF (llama.cpp)
`);
}

async function parseArgs(argv) {
  const args = { file: null, port: 8800, open: true, json: false };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') { printHelp(); process.exit(0); }
    if (arg === '-v' || arg === '--version') {
      const { createRequire } = await import('node:module');
      const require = createRequire(import.meta.url);
      console.log(require('../package.json').version);
      process.exit(0);
    }
    if (arg === '--port') { args.port = parseInt(argv[++i], 10); continue; }
    if (arg === '--no-open') { args.open = false; continue; }
    if (arg === '--json') { args.json = true; continue; }
    if (!arg.startsWith('-')) { args.file = arg; continue; }
    console.error(`Unknown option: ${arg}`);
    process.exit(1);
  }

  return args;
}

function openBrowser(url) {
  const cmd = process.platform === 'darwin' ? 'open'
    : process.platform === 'win32' ? 'start'
    : 'xdg-open';
  exec(`${cmd} ${url}`);
}

async function main() {
  const args = await parseArgs(process.argv);

  if (!args.file) {
    printHelp();
    process.exit(1);
  }

  const filePath = resolve(args.file);

  if (!existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
  }

  const fileName = basename(filePath);
  console.log(`\n  mlpeel — loading ${fileName}\n`);

  const { format, buffer } = await detect(filePath);
  console.log(`  Format:  ${format}`);

  const parser = PARSERS[format];
  if (!parser) {
    console.error(`  Unsupported format: ${format}`);
    process.exit(1);
  }

  const start = performance.now();
  const modelData = parser(buffer);
  const elapsed = (performance.now() - start).toFixed(0);

  console.log(`  Tensors: ${modelData.tensors.length}`);
  console.log(`  Nodes:   ${modelData.graph.nodes.length}`);
  console.log(`  Edges:   ${modelData.graph.edges.length}`);
  console.log(`  Parsed:  ${elapsed}ms\n`);

  if (args.json) {
    console.log(JSON.stringify(modelData, null, 2));
    process.exit(0);
  }

  const { url } = await startServer(modelData, { port: args.port, fileName, modelPath: filePath, format });
  console.log(`  Viewer:  ${url}\n`);

  if (args.open) openBrowser(url);

  console.log('  Press Ctrl+C to stop\n');

  process.on('SIGINT', () => {
    console.log('\n  Shutting down...\n');
    process.exit(0);
  });
}

main().catch((err) => {
  console.error(`\n  Error: ${err.message}\n`);
  process.exit(1);
});
