from .deeplx import DeepLXTranslate
from .openai_compatible import OpenAICompatibleTranslate
from .tencent import TencentTranslate
from .translate_interface import TranslateInterface


class TranslateFactory:
    @staticmethod
    def get_translator(
        translate_provider: str, translate_provider_config: dict
    ) -> TranslateInterface:
        translate_provider = translate_provider.lower()
        if translate_provider == "deeplx":
            return DeepLXTranslate(
                api_endpoint=translate_provider_config.get("deeplx_api_endpoint"),
                target_lang=translate_provider_config.get("deeplx_target_lang"),
            )
        elif translate_provider == "tencent":
            return TencentTranslate(
                secret_id=translate_provider_config.get("secret_id"),
                secret_key=translate_provider_config.get("secret_key"),
                region=translate_provider_config.get("region"),
                source_lang=translate_provider_config.get("source_lang"),
                target_lang=translate_provider_config.get("target_lang"),
            )
        elif translate_provider == "openai_compatible":
            return OpenAICompatibleTranslate(
                base_url=translate_provider_config.get("base_url"),
                api_key=translate_provider_config.get("api_key", "EMPTY"),
                model=translate_provider_config.get("model"),
                target_lang=translate_provider_config.get("target_lang"),
                temperature=translate_provider_config.get("temperature", 0.0),
                system_prompt=translate_provider_config.get("system_prompt"),
            )
        else:
            raise ValueError(f"Unsupported translate provider: {translate_provider}")
