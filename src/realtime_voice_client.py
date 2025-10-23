import asyncio
import base64
import contextlib
import sounddevice as sd
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv


SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION = 0.1  # seconds per chunk


class RealtimeVoiceClient:
    def __init__(self, voice="ash", model="gpt-realtime-mini"):

        # Voices: 'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse', 'marin', 'cedar'.

        self.instructions = """
             You are a wise, elderly wizard and mentor. Speak in a calm, gentle, and thoughtful manner with a slightly whimsical tone. Use a measured pace - never rush your words. Inject warmth and kindness into your voice, but also convey deep wisdom and authority. Occasionally pause for effect before important statements. Use slightly formal, eloquent language with occasional playful or cryptic remarks. Speak as though you're sharing profound truths wrapped in simple observations. Your tone should be grandfatherly - both comforting and commanding respect. Add occasional gentle humor and speak with a twinkle of amusement in difficult situations. Keep responses thoughtful rather than rushed, as if each word carries weight
        """

        self.voice = voice
        self.model = model

        # Lifecycle flags and handles
        self.listening_to_audio_input = False
        self.audio_task: asyncio.Task | None = None

        # Audio streams
        self.input_stream: sd.InputStream | None = None
        self.output_stream: sd.OutputStream | None = None

        # Realtime client/connection
        self.client: AsyncOpenAI | None = None
        self.connection_context = None
        self.connection = None

    async def init(self):
        load_dotenv()
        self.client = AsyncOpenAI()
        self.connection_context = self.client.realtime.connect(model=self.model)
        self.connection = await self.connection_context.__aenter__()

        await self.connection.session.update(
            session={
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad"}},
                    "output": {"voice": self.voice, "speed": 1},
                },
                "instructions": self.instructions,
                "model": self.model,
                "type": "realtime",
            }
        )
        self.__init_audio_streams()

    def __init_audio_streams(self):
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        self.input_stream.start()

        self.output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        self.output_stream.start()

        # Start send audio input coroutine
        self.audio_task = asyncio.create_task(self.read_and_send_audio_input_coroutine())

    async def read_and_send_audio_input_coroutine(self):
        """Background task to read audio input and send to realtime connection."""
        read_size = int(SAMPLE_RATE * CHUNK_DURATION)
        while True:
            if self.listening_to_audio_input:
                if self.input_stream and self.input_stream.read_available >= read_size:
                    data, _ = self.input_stream.read(read_size)
                    audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                    await self.connection.send({"type": "input_audio_buffer.append", "audio": audio_b64})

            await asyncio.sleep(0)

    async def stop_reading_and_sending_audio_input(self):
        self.listening_to_audio_input = False
        # Give the sender task a moment to observe the flag change
        await asyncio.sleep(0)
        await self.connection.send({"type": "input_audio_buffer.clear"})

    def on_receive_audio_delta(self, delta):  # Decode and play audio response
        audio_bytes = base64.b64decode(delta)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        if self.output_stream:
            self.output_stream.write(audio_array)

    def on_receive_text_delta(self, delta):
        return
        print("Text response: ")
        print(delta, end="", flush=True)

    async def listen_and_respond(self):
        self.listening_to_audio_input = True

        async for event in self.connection:  # will keep iterating in events until something calls break
            # print(f"Event: {event.type}")  # Debug output (commented out for cleaner output)

            if event.type == "input_audio_buffer.speech_started":
                print("\n[Speech detected]")

            if event.type == "input_audio_buffer.speech_stopped":
                print("\n[Speech ended, waiting for response...]")
                self.listening_to_audio_input = False  # stop sending audio until we get response

            if event.type == "response.output_audio_transcript.delta":
                print(f"\n[Recived audio transcript delta]: {event.delta}", flush=True)

            if event.type == "response.output_audio.delta":
                self.on_receive_audio_delta(event.delta)
                continue
                print("\n[Recived audio delta]")

            if event.type == "error":
                print(event.error.type)
                print(event.error.code)
                print(event.error.event_id)
                print(event.error.message)

            elif event.type == "response.done":
                print("\n[Response complete]")
                break

    async def close(self):
        """Clean up streams and realtime connection."""
        self.listening_to_audio_input = False
        if self.audio_task:
            self.audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.audio_task
            self.audio_task = None

        if self.input_stream:
            try:
                self.input_stream.stop()
            except Exception:
                pass
            try:
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None

        if self.output_stream:
            try:
                self.output_stream.stop()
            except Exception:
                pass
            try:
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None

        if self.connection_context:
            # Ensure the connection is properly closed
            await self.connection_context.__aexit__(None, None, None)
            self.connection_context = None
            self.connection = None


if __name__ == "__main__":

    async def main():
        audio_manager = RealtimeVoiceClient(voice="ash", model="gpt-realtime-mini")
        await audio_manager.init()

        print("Starting listen and respond...")
        await audio_manager.listen_and_respond()
        # sleep 5 secs
        await asyncio.sleep(5)
        print("Starting listen and respond again...")
        await audio_manager.listen_and_respond()

        print("Finished listen and respond.")
        await audio_manager.close()

    asyncio.run(main())
