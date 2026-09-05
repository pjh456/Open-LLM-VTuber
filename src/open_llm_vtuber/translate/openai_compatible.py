from loguru import logger
from openai import OpenAI

from .translate_interface import TranslateInterface

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text "
    "into the target language. Output only the translation, "
    "without any explanations or comments."
)


class OpenAICompatibleTranslate(TranslateInterface):
    """Translate text via any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "default",
        target_lang: str = "ja",
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.target_lang = target_lang
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.replace(
            "into the target language", f"into {target_lang}"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def translate(self, text: str) -> str:
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            translation = res.choices[0].message.content
            if not translation:
                raise ValueError("Empty translation returned by the model")
            return translation.strip()
        except Exception as e:
            logger.critical(f"Error translating text '{text}'. Error message: {e}")
            raise
