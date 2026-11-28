// Rough VRAM need of a model: its weights (parameter count x bytes per weight) plus about
// 1.5 GB for the CUDA libraries and the decoding work area. An estimate, not a measurement.
const MODEL_PARAMS_M = { 'large-v3-turbo': 809, 'large-v3': 1550, 'alignatt-large-v3-turbo': 809, 'alignatt-large-v3': 1550 };
const WEIGHT_BYTES = { float16: 2, int8_float16: 1, int8: 1, float32: 4 };

// GB; 0 when the model runs on the CPU, null for a model or compute type it doesn't know.
export function estimateVramGb(model, compute, onCpu) {
  if (onCpu) return 0;
  const bytes = WEIGHT_BYTES[compute === 'auto' ? 'float16' : compute];
  if (!MODEL_PARAMS_M[model] || !bytes) return null;
  return (MODEL_PARAMS_M[model] * bytes) / 1024 + 1.5;
}

// "6.6/8GB": memory in use on the whole GPU (every program), as nvidia-smi reports it.
export function vramText(gpu) {
  return `${(gpu.used_mb / 1024).toFixed(1)}/${Math.round(gpu.total_mb / 1024)}GB`;
}
