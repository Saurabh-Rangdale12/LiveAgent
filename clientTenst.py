import asyncio
import websockets
import json
import base64
import wave

# --- Configuration ---
WEBSOCKET_URI = "ws://localhost:8765"
AUDIO_FILE = "download.wav"
IMAGE_FILE = "image.png"
OUTPUT_AUDIO_FILE = "output.wav"

# --- Main Client Logic ---

async def run_test_client():
    """
    Connects to the WebSocket server, sends pre-recorded audio and a static image,
    and logs the responses.
    """
    print("--- Terminal Test Client ---")
    print(f"Connecting to {WEBSOCKET_URI}...")

    received_audio_chunks = []

    try:
        async with websockets.connect(WEBSOCKET_URI, timeout=10) as websocket:
            print("Successfully connected to the server.")

            # Create two concurrent tasks: one for sending, one for receiving.
            # This ensures we are always listening for responses while sending data.
            
            # --- Receiver Task ---
            async def receiver(ws):
                while True:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        if data.get("type") == "ready":
                            print("[SERVER]: Ready signal received. Starting sender...")
                            # Signal the sender to start by closing the queue it's waiting on
                            sender_task.ready.set()

                        elif data.get("type") == "text":
                            print(f"[GEMINI TEXT]: {data.get('data')}")

                        elif data.get("type") == "audio":
                            print("[SERVER]: Received an audio chunk from Gemini.")
                            audio_data = base64.b64decode(data.get("data"))
                            received_audio_chunks.append(audio_data)

                        elif data.get("type") == "turn_complete":
                            print("[SERVER]: Turn Complete.")
                            # You could add logic here to stop the client if needed
                            
                        elif data.get("type") == "interrupted":
                            print("[SERVER]: Response Interrupted.")

                    except websockets.exceptions.ConnectionClosed:
                        print("[INFO]: Connection closed by server.")
                        break
                    except Exception as e:
                        print(f"Error in receiver: {e}")
                        break

            # --- Sender Task ---
            async def sender(ws):
                # An event to wait for the "ready" signal from the receiver
                sender.ready = asyncio.Event()
                await sender.ready.wait()
                
                # 1. Send the image first
                try:
                    with open(IMAGE_FILE, "rb") as f:
                        image_bytes = f.read()
                    b64_image = base64.b64encode(image_bytes).decode('utf-8')
                    await ws.send(json.dumps({"type": "video", "data": b64_image}))
                    print(f"[CLIENT]: Sent image '{IMAGE_FILE}'.")
                except FileNotFoundError:
                    print(f"[ERROR]: Image file not found: {IMAGE_FILE}. Skipping image send.")
            
                # 2. Stream the audio file in chunks
                try:
                    with wave.open(AUDIO_FILE, 'rb') as wf:
                        # Validate audio format
                        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                            print(f"[ERROR]: Audio file '{AUDIO_FILE}' must be 16-bit, 16kHz Mono PCM.")
                            return

                        chunk_size = 4096 # Send audio in 4KB chunks
                        print(f"[CLIENT]: Streaming audio from '{AUDIO_FILE}'...")
                        while True:
                            audio_chunk = wf.readframes(chunk_size // 2) # 2 bytes per frame
                            if not audio_chunk:
                                break # End of file
                            b64_audio = base64.b64encode(audio_chunk).decode('utf-8')
                            await ws.send(json.dumps({"type": "audio", "data": b64_audio}))
                            # Small delay to simulate real-time streaming
                            await asyncio.sleep(0.1) 
                    print("[CLIENT]: Finished sending audio.")

                except FileNotFoundError:
                    print(f"[ERROR]: Audio file not found: {AUDIO_FILE}. Cannot send audio.")

            # Run both tasks concurrently
            receiver_task = asyncio.create_task(receiver(websocket))
            sender_task = asyncio.create_task(sender(websocket))
            
            await asyncio.gather(receiver_task, sender_task)

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError):
        print("\n[ERROR]: Connection failed. Is the server running?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # After the connection closes, save the received audio
        if received_audio_chunks:
            print(f"\nSaving received audio to '{OUTPUT_AUDIO_FILE}'...")
            # Note: The server sends back audio in a format that might not be a standard WAV.
            # We save the raw concatenated data. For proper playback, you might need to know
            # the exact format Gemini sends (e.g., MP3, OGG, or raw PCM).
            # Assuming it's raw data that can be wrapped in a WAV container.
            
            # The ADK typically sends audio in a format that needs decoding.
            # We'll save the raw bytes. You can use a tool like FFMPEG to analyze/convert it.
            # A simple WAV write might fail if the format isn't raw PCM.
            try:
                full_audio_data = b"".join(received_audio_chunks)
                with open(OUTPUT_AUDIO_FILE.replace('.wav', '.bin'), "wb") as f:
                    f.write(full_audio_data)
                print("Saved raw audio data. You may need to convert it for playback.")
            except Exception as e:
                print(f"Could not save audio file: {e}")
        else:
            print("\nNo audio chunks were received from the server.")


if __name__ == "__main__":
    asyncio.run(run_test_client())