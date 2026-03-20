import asyncio
import websockets
import sys
import time
import json
import wave
import numpy as np

def get_audio_data(file_path: str):
    with wave.open(file_path, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        
        duration = frames / float(rate)
        raw_bytes = f.readframes(frames)
        
        # Convert to float32. Assuming 16-bit PCM.
        if sampwidth == 2:
            audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported bit depth: {sampwidth*8}-bit. Please use 16-bit WAV files.")
            
        # Convert stereo to mono
        if channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1)

        # Resample to 16000 Hz if needed
        if rate != 16000:
            print(f"Resampling from {rate}Hz to 16000Hz...")
            import scipy.signal
            new_length = int(len(audio_array) * 16000 / rate)
            audio_array = scipy.signal.resample(audio_array, new_length)
            rate = 16000
            
        # Ensure it's explicitly float32 before converting to bytes
        float_bytes = audio_array.astype(np.float32).tobytes()
        
        return duration, float_bytes, rate, 1

async def stream_audio_and_benchmark(audio_file_path: str, chunk_duration_ms: int = 500):
    uri = "ws://localhost:8000/attranscribe"
    print(f"Connecting to {uri} for streaming benchmark...")
    
    try:
        duration, data, rate, chan = get_audio_data(audio_file_path)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return

    # In a float32 array, each sample is 4 bytes
    bytes_per_sample = 4
    samples_per_chunk = int(rate * (chunk_duration_ms / 1000.0))
    chunk_size_bytes = samples_per_chunk * bytes_per_sample

    print(f"Audio Duration: {duration:.2f} seconds")
    print(f"Simulating streaming with chunk size: {chunk_size_bytes} bytes Float32 ({chunk_duration_ms} ms)")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            
            total_text = ""
            latencies = []
            
            start_streaming_time = time.perf_counter()
            offset = 0
            
            while offset < len(data):
                end_offset = min(offset + chunk_size_bytes, len(data))
                chunk = data[offset:end_offset]
                
                chunk_send_time = time.perf_counter()
                await websocket.send(chunk)
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_time = time.perf_counter()
                    
                    try:
                        result = json.loads(response)
                        text = result.get("Transcription", result.get("text", ""))
                        if text and text.strip():
                            latency = (response_time - chunk_send_time) * 1000
                            latencies.append(latency)
                            total_text += text.strip() + " "
                            print(f"[Chunk {offset//chunk_size_bytes}] Latency: {latency:.2f} ms | Text: {text.strip()}")
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
                
                offset += chunk_size_bytes
                
                elapsed = time.perf_counter() - chunk_send_time
                sleep_time = (chunk_duration_ms / 1000.0) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            total_streaming_time = time.perf_counter() - start_streaming_time
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            wpm = (len(total_text.split()) / total_streaming_time) * 60 if total_streaming_time > 0 else 0

            print("\n" + "="*50)
            print("🌊 STREAMING BENCHMARK RESULTS")
            print("="*50)
            print(f"Audio Duration         : {duration:.2f} seconds")
            print(f"Total Streaming Time   : {total_streaming_time:.2f} seconds")
            print(f"Average Chunk Latency  : {avg_latency:.2f} ms")
            print(f"Max Chunk Latency      : {max_latency:.2f} ms")
            print(f"Word Processing Rate   : {wpm:.2f} WPM")
            print("="*50)
            print(f"Transcribed Text: {total_text.strip()}")
            print("="*50)
            
            resume_latency = int(avg_latency + (50 - avg_latency % 50)) if avg_latency else 200
            print("\n💡 Resume Bullet Point Ideas:")
            print(f"- Designed a streaming audio transcription microservice processing {int(chunk_duration_ms)}ms chunks with an average latency of {avg_latency:.1f}ms per chunk.")
            print(f"- Achieved real-time streaming capabilities with max latency under {int(max_latency + 50)}ms, enabling seamless continuous dictation.")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_stream.py <path_to_wav_file> [chunk_ms]")
        sys.exit(1)
        
    chunk_ms = 500
    if len(sys.argv) >= 3:
        chunk_ms = int(sys.argv[2])
    
    asyncio.run(stream_audio_and_benchmark(sys.argv[1], chunk_ms))
