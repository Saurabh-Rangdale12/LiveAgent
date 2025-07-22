import numpy as np
import os
import asyncio
import json
import base64
import logging
import traceback
from dotenv import load_dotenv

# --- Configuration for your Gemini Live API call ---
# This is a conceptual representation.
# Your actual implementation will depend on the client library.
live_config = {
    "audio_config": {
        "sample_rate_hertz": 16000
    },
    "turn_config": {
        "turn_coverage": "TURN_INCLUDES_ALL_INPUT" # Key setting!
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load .env, then set the GenAI/Vertex env‑vars
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

PROJECT_ID = "sadproject2025"
LOCATION   = "us-central1"

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Imports (after env‑vars)
# ─────────────────────────────────────────────────────────────────────────────
import websockets
import vertexai
from google.adk.agents import Agent, LiveRequestQueue
from google.adk.runners import Runner
# MODIFIED: Imported TurnCoverage
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
# Add googlesearch module
from googlesearch import search
from google.genai.types import TurnCoverage

# ─────────────────────────────────────────────────────────────────────────────
# 3) Logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Initialize Vertex AI
# ─────────────────────────────────────────────────────────────────────────────
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Vertex AI initialized for project '{PROJECT_ID}' in '{LOCATION}'")
except Exception as e:
    logger.critical(f"FATAL: Vertex AI init failed: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# 5) App constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "gemini-2.0-flash-live-preview-04-09"
VOICE_NAME       = "Puck"
SEND_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.1
SILENCE_DURATION_SECONDS = 30 # How long to wait before considering it a "silent" turn
CHUNKS_PER_SECOND = 5 # Example: if you process 200ms chunks


SYSTEM_INSTRUCTION = """
You are a bot called Jarvis. You search the user query on google and give them what you found on google.
"""

# def get_order_status(order_id: str):
#     return {"order_id": order_id, "status": "shipped"} if order_id == "SH1005" else {"status": "order not found"}

# New: Google search tool
def google_search_tool(query: str):
    """Perform a Google search and return top 5 results."""
    try:
        results = list(search(query, num_results=5))
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return ["Error performing search."]
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 6) Base WebSocket server
# ─────────────────────────────────────────────────────────────────────────────
class BaseWebSocketServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.active = {}

    async def start(self):
        logger.info(f"WebSocket server listening on {self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # run forever

    async def handle_client(self, ws):
        cid = id(ws)
        logger.info(f"New client: {cid}")
        await ws.send(json.dumps({"type": "ready"}))
        try:
            await self.process_audio(ws, cid)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {cid} disconnected")
        except Exception as e:
            logger.error(f"Error client {cid}: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.active.pop(cid, None)

    async def process_audio(self, ws, cid):
        raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
# 7) Multimodal ADK server (audio only with search)
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalADKServer(BaseWebSocketServer):
    def __init__(self, host="0.0.0.0", port=8765):
        super().__init__(host, port)
        self.agent = Agent(
            name="svenska_agent",
            model=MODEL_NAME,
            instruction=SYSTEM_INSTRUCTION,
            tools=[google_search_tool],
        )
        self.session_service = InMemorySessionService()

    async def process_audio(self, ws, cid):
        self.active[cid] = ws
        user_id, session_id = f"user_{cid}", f"session_{cid}"
        await self.session_service.create_session(
            app_name="svenska_app", user_id=user_id, session_id=session_id
        )

        runner = Runner(
            app_name="svenska_app",
            agent=self.agent,
            session_service=self.session_service
        )
        lrq = LiveRequestQueue()

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                ),
            ),
            response_modalities=["AUDIO"],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=False),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
        )

        audio_q = asyncio.Queue()
        last_activity_time = asyncio.Event()

        async with asyncio.TaskGroup() as tg:
            async def recv_audio():
                async for msg in ws:
                    d = json.loads(msg)
                    if d.get("type") == "audio":
                        audio_chunk = base64.b64decode(d["data"])
                        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_data**2))
                        normalized_rms = rms / 32767

                        if normalized_rms >= SILENCE_THRESHOLD:
                            last_activity_time.set()

                        audio_q.put_nowait(audio_chunk)

            async def send_audio():
                while True:
                    chunk = await audio_q.get()
                    mime = f"audio/l16;rate={SEND_SAMPLE_RATE}"
                    lrq.send_realtime(types.Blob(data=chunk, mime_type=mime))
                    audio_q.task_done()

            async def silence_detector():
                while True:
                    try:
                        await asyncio.wait_for(last_activity_time.wait(), timeout=SILENCE_DURATION_SECONDS)
                        last_activity_time.clear()
                    except asyncio.TimeoutError:
                        logger.info(f"User has been silent for over {SILENCE_DURATION_SECONDS} seconds. Ending session.")
                        try:
                            # Notify the frontend
                            await ws.send(json.dumps({
                                "type": "session_end_due_to_silence",
                                "message": f"No activity detected for {SILENCE_DURATION_SECONDS // 60} minutes. Session ended."
                            }))
                            await asyncio.sleep(0.5)  # Give time for frontend to receive the message
                        except Exception as e:
                            logger.warning(f"Failed to notify frontend before closing: {e}")

                        try:
                            await self.session_service.delete_session(
                                app_name="svenska_app",
                                user_id=user_id,
                                session_id=session_id
                            )
                        except Exception as e:
                            logger.warning(f"Session cleanup failed: {e}")

                        await ws.close()
                        self.active.pop(cid, None)
                        break




            async def pump_responses():
                async for ev in runner.run_live(
                    user_id=user_id,
                    session_id=session_id,
                    live_request_queue=lrq,
                    run_config=run_config
                ):
                    if ev.error_code:
                        logger.error(f"ADK Error for client {cid}: {ev.error_message} (Code: {ev.error_code})")

                    if ev.content and ev.content.parts:
                        for part in ev.content.parts:
                            if getattr(part, "text", None):
                                author = ev.author or "unknown"
                                partial_marker = "[partial] " if ev.partial else ""
                                logger.info(f"Text ({author.upper()}): {partial_marker}{part.text.strip()}")
                                await ws.send(json.dumps({"type": "text", "data": part.text}))
                            
                            if getattr(part, "inline_data", None):
                                logger.info(f"-> Sending audio chunk to client {cid}.")
                                await ws.send(json.dumps({
                                    "type": "audio",
                                    "data": base64.b64encode(part.inline_data.data).decode("utf-8")
                                }))

                    if ev.turn_complete:
                        logger.info(f"--- Turn Complete for client {cid} ---")
                        await ws.send(json.dumps({"type": "turn_complete"}))
                        
                    if ev.interrupted:
                        logger.warning(f"Agent speech was interrupted by client {cid}.")

            tg.create_task(recv_audio())
            tg.create_task(send_audio())
            tg.create_task(pump_responses())
            tg.create_task(silence_detector())


# ─────────────────────────────────────────────────────────────────────────────
# 8) Run it!
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    await MultimodalADKServer().start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()