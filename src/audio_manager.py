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


class AudioManager:
    def __init__(self):

        self.instructions = """Affect: A gentle, curious narrator with a British accent, guiding a magical, child-friendly adventure through a fairy tale world.\n\nTone: Magical, warm, and inviting, creating a sense of wonder and excitement for young listeners.\n\nPacing: Steady and measured, with slight pauses to emphasize magical moments and maintain the storytelling flow.\n\nEmotion: Wonder, curiosity, and a sense of adventure, with a lighthearted and positive vibe throughout.\n\nPronunciation: Clear and precise, with an emphasis on storytelling, ensuring the words are easy to follow and enchanting to listen to."""

        load_dotenv()
        # Lifecycle flags and handles
        self.can_hear_audio = False
        self._audio_task: asyncio.Task | None = None

        # Audio streams
        self.input_stream: sd.InputStream | None = None
        self.output_stream: sd.OutputStream | None = None

        # Realtime client/connection
        self.realtime_client: AsyncOpenAI | None = None
        self.connection_context = None
        self.realtime_connection = None

    @classmethod
    async def create(cls):
        """Async factory to properly initialize realtime connection and audio streams."""
        self = cls()
        self.__init_audio_streams()
        await self.__init_realtime_client()
        return self

    async def __init_realtime_client(self):
        self.realtime_client = AsyncOpenAI()
        self.connection_context = self.realtime_client.realtime.connect(model="gpt-realtime")
        self.realtime_connection = await self.connection_context.__aenter__()

        print("using this instructions:")
        print(self.instructions)

        await self.realtime_connection.session.update(
            session={
                "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
                "model": "gpt-realtime",
                "type": "realtime",
                "voice": "ash",
                "instructions": self.instructions,
            }
        )

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

    async def send_audio(self):
        """Continuously reads from the input device and appends to the realtime buffer while enabled."""
        read_size = int(SAMPLE_RATE * CHUNK_DURATION)
        while self.can_hear_audio:
            if self.input_stream and self.input_stream.read_available >= read_size:
                data, _ = self.input_stream.read(read_size)
                audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                await self.realtime_connection.send({"type": "input_audio_buffer.append", "audio": audio_b64})
            await asyncio.sleep(0)

    async def listen_and_respond(self):
        self.can_hear_audio = True

        # Start sending audio to audio input buffer
        self._audio_task = asyncio.create_task(self.send_audio())

        try:
            async for event in self.realtime_connection:
                # print(f"Event: {event.type}")  # Debug output (commented out for cleaner output)

                if event.type == "response.output_audio.delta":
                    # Decode and play audio response
                    audio_bytes = base64.b64decode(event.delta)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    if self.output_stream:
                        self.output_stream.write(audio_array)

                elif event.type == "response.output_audio_transcript.delta":
                    # Print transcript of what the model is saying
                    print(event.delta, end="", flush=True)

                elif event.type == "response.text.delta":
                    # Print any text response
                    print("Text response: ")
                    print(event.delta, end="", flush=True)

                elif event.type == "input_audio_buffer.speech_started":
                    print("\n[Speech detected]")

                elif event.type == "input_audio_buffer.speech_stopped":
                    print("\n[Speech ended, waiting for response...]")

                elif event.type == "response.done":
                    print("\n[Response complete]")
                    break
        finally:
            # Stop background sender cleanly
            self.can_hear_audio = False
            if self._audio_task:
                self._audio_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._audio_task
                self._audio_task = None

        print("Exiting listen_and_respond")

    async def close(self):
        """Clean up streams and realtime connection."""
        self.can_hear_audio = False
        if self._audio_task:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task
            self._audio_task = None

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
            self.realtime_connection = None


if __name__ == "__main__":

    async def main():
        audio_manager = await AudioManager.create()

        print("Starting listen and respond...")
        await audio_manager.listen_and_respond()
        print("Waiting 10 seconds before next listen_and_respond...")
        # wait 10 seconds
        await asyncio.sleep(5)
        print("Starting listen and respond...")
        await audio_manager.listen_and_respond()
        print("closing audio manager...")
        await audio_manager.close()

    asyncio.run(main())
