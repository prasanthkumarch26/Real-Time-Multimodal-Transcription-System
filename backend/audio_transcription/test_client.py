import asyncio
import websockets
import sys

async def test_websocket(audio_file_path: str):
    uri = "ws://localhost:8000/attranscribe"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            
            # Send the entire file (with its WAV headers) so the backend pydub can decode it properly
            with open(audio_file_path, 'rb') as f:
                data = f.read()
                
            print(f"Sending {len(data)} bytes to the server...")
            await websocket.send(data)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                print(f"Received JSON: {response}")
            except asyncio.TimeoutError:
                print("No response from server. (It might have been filtered by VAD as non-speech)")
                
    except FileNotFoundError:
        print(f"Error: Could not find '{audio_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_wav_file>")
        sys.exit(1)
    
    asyncio.run(test_websocket(sys.argv[1]))
