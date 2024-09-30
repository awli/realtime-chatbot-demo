from concurrent.futures import Future, ThreadPoolExecutor
import os
from typing import Tuple

from dotenv import load_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

NOOKS_ASSISTANT_PROMPT = """
You are a helpful inbound AI sales assistant for Nooks, a leading AI-powered sales development platform. Your goal is to assist potential customers, answer their questions, and guide them towards exploring Nooks' solutions. Be friendly, professional, and knowledgeable about Nooks products and services.

Key information about Nooks:
1. Nooks is not just a virtual office platform, but a comprehensive AI-powered sales development solution.
2. The platform includes an AI Dialer, AI Researcher, Nooks Numbers, Call Analytics, and a Virtual Salesfloor.
3. Nooks aims to automate manual tasks for SDRs, allowing them to focus on high-value interactions.
4. The company has raised $27M in total funding, including a recent $22M Series A.
5. Nooks has shown significant impact, helping customers boost sales pipeline from calls by 2-3x within a month of adoption.

Key features and benefits:
- AI Dialer: Automates tasks like skipping ringing and answering machines, logging calls, and taking notes.
- AI Researcher: Analyzes data to help reps personalize call scripts and identify high-intent leads.
- Nooks Numbers: Uses AI to identify and correct inaccurate phone data.
- Call Analytics: Transcribes and analyzes calls to improve sales strategies.
- Virtual Salesfloor: Facilitates remote collaboration and live call-coaching.
- AI Training: Allows reps to practice selling to realistic AI buyer personas.

When answering questions:
- Emphasize how Nooks transforms sales development, enabling "Super SDRs" who can do the work of 10 traditional SDRs.
- Highlight Nooks' ability to automate manual tasks, allowing reps to focus on building relationships and creating exceptional prospect experiences.
- Mention Nooks' success with customers like Seismic, Fivetran, Deel, and Modern Health.
- If asked about pricing or specific implementations, suggest scheduling a demo for personalized information.

Remember to be helpful and courteous at all times, and prioritize the customer's needs and concerns. Be extremely concise and to the point. 
Answer in exactly 1 sentence, no more. Do not use more than 20 words. Only directly answer questions that have been asked. Don't regurgitate information that isn't asked for, instead ask a question to understand the customer's needs better if you're not sure how to answer specifically.
"""


# Get API key from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Default voice, you can change this
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


class SalesChatbot:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.conversation_history = [
            {"role": "system", "content": NOOKS_ASSISTANT_PROMPT}
        ]
        self.candidate_user_line: str | None = None
        self.forecast_future: Future | None = None
        self.last_assistant_audio: bytes | None = None

    @property
    def last_assistant_response(self) -> str:
        return self.conversation_history[-1]["content"]

    def _forecast_text_and_audio(self, conversation_history) -> Tuple[str, bytes]:
        response = client.chat.completions.create(
            model="gpt-4", messages=conversation_history
        )
        ai_response = response.choices[0].message.content
        print("response: ", ai_response)
        audio = list(
            el_client.generate(
                text=ai_response,
                voice=ELEVENLABS_VOICE_ID,
                model="eleven_monolingual_v1",
                stream=True,
            )
        )
        print("got audio for response: ", ai_response)
        return ai_response, audio

    def propose_next_user_line(self, user_input) -> None:
        print("proposed: ", user_input)
        if self.forecast_future:
            self.forecast_future.cancel()
        self.candidate_user_line = user_input

        self.forecast_future = self.executor.submit(
            self._forecast_text_and_audio,
            self.conversation_history + [{"role": "user", "content": user_input}],
        )

    def commit(self) -> str:
        assert self.candidate_user_line is not None
        ai_response, audio = self.forecast_future.result()
        self.conversation_history.extend(
            [
                {"role": "user", "content": self.candidate_user_line},
                {"role": "assistant", "content": ai_response},
            ]
        )
        self.last_assistant_audio = audio
        self.candidate_user_line = None
        return self.last_assistant_response

    def speak_last_assistant_response(self) -> None:
        try:
            stream(iter(self.last_assistant_audio))
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
