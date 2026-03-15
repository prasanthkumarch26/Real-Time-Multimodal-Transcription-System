/**
 * audio-processor.js — AudioWorklet Processor
 *
 * This runs in a dedicated audio rendering thread (not the main thread).
 * It accumulates incoming Float32 PCM samples at 16kHz mono and
 * sends batches back to the main thread every ~30ms.
 *
 * Main thread forwards these binary Float32Array chunks to the backend
 * via WebSocket. This gives <50ms end-to-end audio latency vs. MediaRecorder.
 *
 * NOTE: This file must be served from /public so it can be loaded via
 *   audioContext.audioWorklet.addModule('/audio-processor.js')
 */

const FLUSH_INTERVAL_SAMPLES = 480; // 30ms at 16kHz (480 samples × (1/16000s) = 0.03s)

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
  }

  /**
   * process() is called by the audio rendering thread with each 128-sample block.
   * We accumulate blocks until we have FLUSH_INTERVAL_SAMPLES worth,
   * then post the Float32Array to the main thread.
   */
  process(inputs, _outputs, _parameters) {
    const input = inputs[0];

    // input[0] = mono channel (we configure mono on AudioContext side)
    if (input && input.length > 0) {
      const channelData = input[0]; // Float32Array of 128 samples

      for (let i = 0; i < channelData.length; i++) {
        this._buffer.push(channelData[i]);
      }

      if (this._buffer.length >= FLUSH_INTERVAL_SAMPLES) {
        // Transfer ownership of the buffer to the main thread — zero-copy
        const chunk = new Float32Array(this._buffer.splice(0, FLUSH_INTERVAL_SAMPLES));
        this.port.postMessage(chunk, [chunk.buffer]);
      }
    }

    // Return true to keep the processor alive
    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
