import streamlit as st
import asyncio
import websockets
import base64
import json
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- Configuration ---
WEBSOCKET_URI = "http://localhost:8765"

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = asyncio.Queue()
if "video_queue" not in st.session_state:
    st.session_state.video_queue = asyncio.Queue()
if "text_messages" not in st.session_state:
    st.session_state.text_messages = []
if "status_message" not in st.session_state:
    st.session_state.status_message = "Press 'Start' to begin"
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

# --- WebSocket Communication ---

# This function runs in a separate thread to handle WebSocket communication
def websocket_thread_func(audio_queue, video_queue, text_messages, audio_to_play, status_message_state):
    
    async def main_async():
        try:
            async with websockets.connect(WEBSOCKET_URI) as websocket:
                status_message_state["text"] = "Connected to server..."

                # Coroutine to send data from queues to WebSocket
                async def sender():
                    while st.session_state.is_running:
                        try:
                            # Send audio data
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                            b64_audio = base64.b64encode(audio_data).decode('utf-8')
                            await websocket.send(json.dumps({"type": "audio", "data": b64_audio}))

                            # Send video data (if available)
                            if not video_queue.empty():
                                video_frame = await video_queue.get()
                                b64_video = base64.b64encode(video_frame).decode('utf-8')
                                await websocket.send(json.dumps({"type": "video", "data": b64_video}))

                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"Sender error: {e}")
                            break

                # Coroutine to receive data from WebSocket
                async def receiver():
                    while st.session_state.is_running:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            if data['type'] == 'text':
                                text_messages.append(("Gemini", data['data']))
                            elif data['type'] == 'audio':
                                audio_to_play["data"] = data['data']
                            elif data['type'] == 'turn_complete':
                                status_message_state["text"] = "Gemini is done. Your turn!"
                            elif data['type'] == 'interrupted':
                                text_messages.append(("System", "Interrupted by user."))
                            elif data['type'] == 'ready':
                                 status_message_state["text"] = "Ready! Speak now."

                        except websockets.exceptions.ConnectionClosed:
                            status_message_state["text"] = "Connection closed."
                            st.session_state.is_running = False
                            break
                        except Exception as e:
                            print(f"Receiver error: {e}")
                            break
                
                # Run sender and receiver concurrently
                await asyncio.gather(sender(), receiver())

        except Exception as e:
            status_message_state["text"] = f"Failed to connect: {e}"
            st.session_state.is_running = False

    # Run the asyncio event loop
    asyncio.run(main_async())


# --- Audio Processing for streamlit-webrtc ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = st.session_state.audio_queue

    async def recv_async(self, frame):
        # We need to convert the audio frame to raw PCM data (16-bit integers)
        # This is a simplification. A more robust solution would handle format conversion.
        resampled_frame = frame.resample(sample_rate=16000)
        pcm_s16le = (resampled_frame.to_ndarray() * 32767).astype("int16").tobytes()
        if st.session_state.is_running:
            await self.audio_queue.put(pcm_s16le)
        return frame, {} # Return the frame to keep the stream going

# --- Streamlit UI ---

st.set_page_config(layout="wide")

st.title("Live Conversation with Gemini (Streamlit)")

col1, col2 = st.columns([1, 1])

with col1:
    # --- Video and Controls ---
    st.header("Your Camera")
    
    # The WebRTC component that captures audio and video
    webrtc_ctx = webrtc_streamer(
        key="webrtc-streamer",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

    # --- Start / Stop Button ---
    if st.button("Start Conversation", disabled=st.session_state.is_running):
        st.session_state.is_running = True
        st.session_state.text_messages = []
        status_message_state = {"text": "Connecting..."}
        audio_to_play_state = {"data": None}
        
        # We use these dicts to pass state to the thread because Streamlit
        # can't directly share st.session_state across threads safely.
        st.session_state.status_message = status_message_state
        st.session_state.audio_to_play = audio_to_play_state
        
        # Start the WebSocket communication in a background thread
        thread = threading.Thread(
            target=websocket_thread_func,
            args=(
                st.session_state.audio_queue,
                st.session_state.video_queue,
                st.session_state.text_messages,
                st.session_state.audio_to_play,
                st.session_state.status_message,
            ),
        )
        thread.start()
        st.rerun()

    if st.button("End Conversation", disabled=not st.session_state.is_running):
        st.session_state.is_running = False
        st.rerun()

    # --- Status Display ---
    st.info(st.session_state.status_message.get("text", "Press 'Start' to begin"))
    
    # --- Audio Playback ---
    if st.session_state.audio_to_play and st.session_state.audio_to_play.get("data"):
        audio_base64 = st.session_state.audio_to_play["data"]
        # Use a unique key to force re-rendering and autoplay
        audio_html = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}" class="stAudio"></audio>'
        st.components.v1.html(audio_html, height=0)
        # Clear the audio data after playing
        st.session_state.audio_to_play["data"] = None


with col2:
    # --- Transcription Display ---
    st.header("Conversation Log")
    chat_container = st.container(height=500)
    with chat_container:
        for author, message in st.session_state.text_messages:
            with st.chat_message(author):
                st.write(message)