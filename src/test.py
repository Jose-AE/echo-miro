import asyncio
import base64
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from openai import AsyncOpenAI

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION = 0.1  # seconds per chunk


async def main():
    load_dotenv()
    client = AsyncOpenAI()

    print("Connecting to Realtime API...")

    conn_cm = client.realtime.connect(model="gpt-realtime")
    conn = await conn_cm.__aenter__()

    try:
        # Use server VAD to automatically detect end of speech
        await conn.session.update(
            session={
                "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
                "model": "gpt-realtime",
                "type": "realtime",
            }
        )

        print("Speak now... (model will respond automatically when you stop)")

        # Start audio input stream
        read_size = int(SAMPLE_RATE * CHUNK_DURATION)
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        stream.start()

        async def send_audio():
            while True:
                if stream.read_available >= read_size:
                    data, _ = stream.read(read_size)
                    audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                    await conn.send({"type": "input_audio_buffer.append", "audio": audio_b64})
                await asyncio.sleep(0)

        # Start sending audio in background
        audio_task = asyncio.create_task(send_audio())

        # Start audio output stream for playing response
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        output_stream.start()

        # Wait for LLM response
        try:
            async for event in conn:
                # print(f"Event: {event.type}")  # Debug output (commented out for cleaner output)

                if event.type == "response.output_audio.delta":
                    # Decode and play audio response
                    audio_bytes = base64.b64decode(event.delta)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    output_stream.write(audio_array)

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
            audio_task.cancel()
            stream.stop()
            stream.close()
            output_stream.stop()
            output_stream.close()
    finally:
        # Ensure the connection is properly closed without using 'with'
        await conn_cm.__aexit__(None, None, None)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
