// Runs in an AudioContext created at 16 kHz: collects mono samples into
// 100 ms chunks and posts them to the main thread as Int16 PCM.
const CHUNK = 1600;

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.len = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    const ch1 = input[1];
    for (let i = 0; i < ch0.length; i++) {
      // Downmix stereo to mono.
      this.buf[this.len++] = ch1 ? (ch0[i] + ch1[i]) * 0.5 : ch0[i];
      if (this.len === CHUNK) this.flush();
    }
    return true;
  }

  flush() {
    const pcm = new Int16Array(CHUNK);
    let sum = 0;
    for (let i = 0; i < CHUNK; i++) {
      const s = Math.max(-1, Math.min(1, this.buf[i]));
      sum += s * s;
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage({ pcm, rms: Math.sqrt(sum / CHUNK) }, [pcm.buffer]);
    this.len = 0;
  }
}

registerProcessor('pcm-capture', PcmCapture);
