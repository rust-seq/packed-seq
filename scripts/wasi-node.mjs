// Run a wasm32-wasip1 binary under Node's V8 engine, for use as a cargo runner:
//
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings scripts/wasi-node.mjs" \
//   RUSTFLAGS="-C target-feature=+simd128" \
//   cargo test --release --features scalar --target wasm32-wasip1
import { readFile } from 'node:fs/promises';
import process from 'node:process';
import { WASI } from 'node:wasi';

const [wasmPath, ...args] = process.argv.slice(2);
if (!wasmPath) {
  console.error('usage: node wasi-node.mjs <file.wasm> [args...]');
  process.exit(2);
}

const wasi = new WASI({
  version: 'preview1',
  args: [wasmPath, ...args],
  env: process.env,
  returnOnExit: true,
});

const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
