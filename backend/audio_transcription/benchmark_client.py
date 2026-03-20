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

async def run_benchmark(audio_file_path: str):
    uri = "ws://localhost:8000/attranscribe"
    print(f"Connecting to {uri}...")
    
    try:
        duration, data, rate, chan = get_audio_data(audio_file_path)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return

    print(f"Audio Duration: {duration:.2f} seconds")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            
            print(f"Sending {len(data)} bytes (Float32 PCM) to the server...")
            
            start_time = time.perf_counter()
            await websocket.send(data)
            
            total_text = ""
            first_token_time = None
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    elapsed_since_send = time.perf_counter() - start_time
                    
                    if first_token_time is None:
                        first_token_time = elapsed_since_send
                        print(f"Time to First Token (TTFT): {first_token_time*1000:.2f} ms")
                    
                    try:
                        result = json.loads(response)
                        if "Transcription" in result:
                            # From main.py, it uses "Transcription" key, not "text"
                            total_text += result["Transcription"] + " "
                        elif "text" in result:
                            total_text += result["text"] + " "
                    except json.JSONDecodeError:
                        total_text += response + " "
                        
                    break
                    
                except asyncio.TimeoutError:
                    print("No response from server. (It might have been filtered by VAD as non-speech)")
                    break
            
            end_time = time.perf_counter()
            total_processing_time = end_time - start_time
            
            rtf = total_processing_time / duration if duration > 0 else 0
            wpm = (len(total_text.split()) / total_processing_time) * 60 if total_processing_time > 0 else 0

            print("\n" + "="*50)
            print("📊 BENCHMARK RESULTS (For Resume/Portfolio)")
            print("="*50)
            print(f"Audio Duration      : {duration:.2f} seconds")
            print(f"Total Processing    : {total_processing_time:.2f} seconds")
            print(f"Time to First Token : {first_token_time * 1000 if first_token_time else 0:.2f} ms")
            print(f"End-to-End Latency  : {total_processing_time * 1000:.2f} ms")
            print(f"Real-Time Factor    : {rtf:.3f} (Lower is better. < 1.0 means faster than real-time)")
            print(f"Word Processing Rate: {wpm:.2f} WPM")
            print("="*50)
            print(f"Transcribed Text: {total_text.strip()}")
            print("="*50)
            
            ttft_ms = int((first_token_time or total_processing_time) * 1000)
            print("\n💡 Resume Bullet Point Ideas:")
            print(f"- Engineered a real-time multimodal transcription system achieving sub-{ttft_ms + (50 - ttft_ms % 50)}ms latency using FastAPI and WebSockets.")
            print(f"- Optimized Faster-Whisper transcription pipeline, reaching an impressive Real-Time Factor (RTF) of {rtf:.2f}.")
            print(f"- Achieved high-throughput streaming transcription capable of processing {int(wpm)} words per minute.")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_client.py <path_to_wav_file>")
        sys.exit(1)
    
    asyncio.run(run_benchmark(sys.argv[1]))
