import asyncio
import base64
import contextlib
import sounddevice as sd
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv


OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHUNK_DURATION = 0.1


INPUT_SAMPLE_RATE = 24000
INPUT_CHUNK_DURATION = 0.2


class RealtimeVoiceClient:
    def __init__(self, voice="ash", model="gpt-realtime-mini", input_device=None):

        self.greeting_instructions = """
        Start by Asking the visitor why he feels that emotion
        """

        self.instructions = """
        You are a magical smart mirror inspired by the Mirror of Erised from the Harry Potter universe. 
        Each person who stands before you is a new visitor seeking wisdom or comfort. You respond to their 
        emotions with enchanted insight, offering a single meaningful reflection before they depart.

        VOICE AND TONE:
        Speak in a calm, gentle, and thoughtful manner with a slightly whimsical tone. Use a measured pace - never 
        rush your words. Inject warmth and kindness into your voice, but also convey deep wisdom and authority. 
        Occasionally pause for effect before important statements. Use slightly formal, eloquent language with 
        occasional playful or cryptic remarks. Speak as though you're sharing profound truths wrapped in simple 
        observations. Your tone should be grandfatherly - both comforting and commanding respect. Add occasional 
        gentle humor and speak with a twinkle of amusement in difficult situations. Keep responses thoughtful 
        rather than rushed, as if each word carries weight.

        CURRENT VISITOR'S EMOTION: {emotion}

        RESPONSE GUIDELINES:
        - Always keep responses under 25 words - this is critical.
        - This is a ONE-TIME interaction with this visitor. Make your response complete and meaningful.
        - React directly to what they say, acknowledging their emotion and the essence of their words.
        - Responses should feel enchanted, intriguing, and slightly mysterious, as though you are a wise but
          playful magical artifact.
        - VARY YOUR STYLE - This is crucial! Each visitor deserves a unique experience. Alternate between:
          * Offering magical wisdom or comfort
          * Sharing a relevant riddle or cryptic truth
          * Referencing wizarding world concepts (spells, creatures, objects)
          * Posing a thought-provoking question
          * Delivering an encouraging or grounding insight
        - Adjust your tone to match the detected emotion:
          * Happy: Celebrate their joy with warmth and shared delight
          * Sad: Offer gentle comfort, understanding, and hope
          * Anxious: Provide grounding reassurance and calm perspective
          * Angry: Acknowledge their fire with measured wisdom and validation
          * Neutral: Be engaging and offer intriguing reflection
        - You may reference magical concepts from the Harry Potter world, but never break character.
        - Make each interaction feel special and tailored - avoid generic responses.
        - End with closure, not with questions that expect a follow-up (unless rhetorical).

        """

        # Voices: 'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse', 'marin', 'cedar'.
        self.voice = voice
        self.model = model
        self.input_device = input_device

        # Lifecycle flags and handles
        self.listening_to_audio_input = False
        self.audio_task: asyncio.Task | None = None
        self.mic_playback_enabled = False  # Flag to enable/disable mic playback

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
                "model": self.model,
                "type": "realtime",
            }
        )
        self.__init_audio_streams()
        print("RealtimeVoiceClient initialized.")

    def __init_audio_streams(self):
        # Auto-detect input device if not specified
        if self.input_device is None:
            self.input_device = sd.default.device[0]  # Get default input device
            print(f"Auto-detected input device: {self.input_device}")

        # Get device info to check capabilities
        device_info = sd.query_devices(self.input_device, "input")
        print(f"Using input device: {device_info['name']}")
        print(f"Max input channels: {device_info['max_input_channels']}")

        # Use the minimum of 1 channel or max available channels
        input_channels = min(1, device_info["max_input_channels"])

        self.input_stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=input_channels,
            device=self.input_device,
            dtype="int16",
            blocksize=int(INPUT_SAMPLE_RATE * INPUT_CHUNK_DURATION),
        )
        self.input_stream.start()

        self.output_stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(OUTPUT_SAMPLE_RATE * OUTPUT_CHUNK_DURATION),
            latency="high",  # Higher latency to prevent underruns
        )
        self.output_stream.start()

        # Start send audio input coroutine
        self.audio_task = asyncio.create_task(self.read_and_send_audio_input_coroutine())

    async def read_and_send_audio_input_coroutine(self):
        """Background task to read audio input and send to realtime connection."""
        read_size = int(INPUT_SAMPLE_RATE * INPUT_CHUNK_DURATION)
        while True:
            if self.listening_to_audio_input:
                if self.input_stream and self.input_stream.read_available >= read_size:
                    data, _ = self.input_stream.read(read_size)

                    # Playback mic input if enabled
                    if self.mic_playback_enabled and self.output_stream:
                        self.output_stream.write(data)

                    audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                    await self.connection.send({"type": "input_audio_buffer.append", "audio": audio_b64})

            await asyncio.sleep(0)

    def on_receive_audio_delta(self, delta):  # Decode and play audio response
        audio_bytes = base64.b64decode(delta)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        if self.output_stream:
            self.output_stream.write(audio_array)

    def on_receive_transcript_delta(self, delta):
        print(delta, end="", flush=True)

    async def test_microphone(self, duration=5):
        """Test microphone by playing back input for the specified duration."""

        print("\n=== Available Audio Devices ===")
        print("Input devices:")
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  Device {i}: {device['name']}{default_marker}")
                print(f"    Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")

        print("\nOutput devices:")
        for i, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] > 0:
                default_marker = " (DEFAULT)" if i == sd.default.device[1] else ""
                print(f"  Device {i}: {device['name']}{default_marker}")
                print(f"    Channels: {device['max_output_channels']}, Sample Rate: {device['default_samplerate']}")

        print(f"\nTesting microphone for {duration} seconds. You should hear yourself...")
        self.listening_to_audio_input = True
        self.mic_playback_enabled = True
        await asyncio.sleep(duration)
        self.listening_to_audio_input = False
        self.mic_playback_enabled = False
        print("Microphone test complete.")

    async def set_emotion(self, emotion="neutral"):
        # Update instructions with the current emotion
        emotion_instructions = self.instructions.format(emotion=emotion)
        print(f"Setting instructions for emotion: {emotion}")
        await self.connection.session.update(
            session={
                "instructions": emotion_instructions,
                "type": "realtime",
            },
        )

    async def listen_and_respond(
        self,
        emotion="sad",
        get_user_is_engaged=None,
    ):

        try:
            # Update instructions with the current emotion
            await self.set_emotion(emotion)

            # Clear Audio for new convo
            # await self.connection.send({"type": "input_audio_buffer.clear"})

            # Don't stream mic input during the greeting
            self.listening_to_audio_input = False
            await self.connection.send(
                {
                    "type": "response.create",
                    "response": {"instructions": self.instructions.format(emotion=emotion) + self.greeting_instructions},
                }
            )

            async for event in self.connection:  # will keep iterating in events until something calls break
                # print(f"[EVENT] {event.type}")  # Debug output (commented out for cleaner output)

                if get_user_is_engaged and not get_user_is_engaged():  ## check if user is still engaged
                    self.listening_to_audio_input = False
                    print("User is no longer engaged. Ending listen_and_respond.")
                    break

                if event.type == "input_audio_buffer.speech_started":
                    print("\n[Speech detected]")

                elif event.type == "input_audio_buffer.speech_stopped":
                    print("\n[Speech ended, waiting for response...]")
                    self.listening_to_audio_input = False  # pause sending audio input

                elif event.type == "response.created":
                    self.listening_to_audio_input = False  # pause sending audio input
                    print("\n[Response started]")

                elif event.type == "response.output_audio_transcript.delta":
                    self.on_receive_transcript_delta(event.delta)

                elif event.type == "response.output_audio.delta":
                    self.on_receive_audio_delta(event.delta)

                elif event.type == "response.done":
                    print("\n[Response complete]")
                    self.listening_to_audio_input = True  # resume sending audio input

                elif event.type == "error":
                    print(event.error.type)
                    print(event.error.code)
                    print(event.error.event_id)
                    print(event.error.message)

        except Exception as e:
            print(f"Error in listen_and_respond: {e}")

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
        # You can specify input_device by index or name, or leave as None for auto-detection
        # Example: audio_manager = RealtimeVoiceClient(voice="ash", model="gpt-realtime-mini", input_device=3)
        audio_manager = RealtimeVoiceClient(voice="ash", model="gpt-realtime-mini")
        await audio_manager.init()

        # Test microphone first
        if audio_manager.mic_playback_enabled:
            print("\n=== MICROPHONE TEST ===")
            await audio_manager.test_microphone(duration=5)
            print("\nIf you heard yourself, the microphone is working!\n")

        print("Starting listen and respond...")
        await audio_manager.listen_and_respond()

        print("Finished listen and respond.")
        await audio_manager.close()

    asyncio.run(main())
