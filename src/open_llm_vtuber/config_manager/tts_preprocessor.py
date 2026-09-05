# config_manager/translate.py
from typing import Literal, Optional, Dict, ClassVar
from pydantic import ValidationInfo, Field, model_validator
from .i18n import I18nMixin, Description

# --- Sub-models for specific Translator providers ---


class DeepLXConfig(I18nMixin):
    """Configuration for DeepLX translation service."""

    deeplx_target_lang: str = Field(..., alias="deeplx_target_lang")
    deeplx_api_endpoint: str = Field(..., alias="deeplx_api_endpoint")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "deeplx_target_lang": Description(
            en="Target language code for DeepLX translation",
            zh="DeepLX 翻译的目标语言代码",
        ),
        "deeplx_api_endpoint": Description(
            en="API endpoint URL for DeepLX service", zh="DeepLX 服务的 API 端点 URL"
        ),
    }


class TencentConfig(I18nMixin):
    """Configuration for tencent translation service."""

    secret_id: str = Field(..., description="Tencent Secret ID")
    secret_key: str = Field(..., description="Tencent Secret Key")
    region: str = Field(..., description="Region for Tencent Service")
    source_lang: str = Field(
        ..., description="Source language code for tencent translation"
    )
    target_lang: str = Field(
        ..., description="Target language code for tencent translation"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "secret_id": Description(en="Tencent Secret ID", zh="腾讯服务的Secret ID"),
        "secret_key": Description(en="Tencent Secret Key", zh="腾讯服务的Secret Key"),
        "region": Description(en="Region for Tencent Service", zh="腾讯服务使用的区域"),
        "source_lang": Description(
            en="Source language code for tencent translation", zh="腾讯翻译的源语言代码"
        ),
        "target_lang": Description(
            en="Target language code for tencent translation",
            zh="腾讯翻译的目标语言代码",
        ),
    }


class OpenAICompatibleConfig(I18nMixin):
    """Configuration for OpenAI-compatible translation service."""

    base_url: str = Field(
        ..., alias="base_url", description="Base URL for the OpenAI-compatible API"
    )
    api_key: str = Field(
        "EMPTY", alias="api_key", description="API key for authentication"
    )
    model: str = Field(..., alias="model", description="Name of the translation model")
    target_lang: str = Field(
        ..., alias="target_lang", description="Target language for translation"
    )
    temperature: float = Field(
        0.0, alias="temperature", description="Sampling temperature for the model"
    )
    system_prompt: str | None = Field(
        None,
        alias="system_prompt",
        description="Custom system prompt (optional, defaults to a generic translation prompt)",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for the OpenAI-compatible API",
            zh="OpenAI 兼容 API 的基础 URL",
        ),
        "api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "model": Description(en="Name of the translation model", zh="翻译模型的名称"),
        "target_lang": Description(
            en="Target language for translation", zh="翻译的目标语言"
        ),
        "temperature": Description(
            en="Sampling temperature for the model", zh="模型的采样温度"
        ),
        "system_prompt": Description(
            en="Custom system prompt (optional)",
            zh="自定义系统提示词（可选）",
        ),
    }


# --- Main TranslatorConfig model ---


class TranslatorConfig(I18nMixin):
    """Configuration for translation services."""

    translate_audio: bool = Field(..., alias="translate_audio")
    translate_provider: Literal["deeplx", "tencent", "openai_compatible"] = Field(
        ..., alias="translate_provider"
    )
    deeplx: Optional[DeepLXConfig] = Field(None, alias="deeplx")
    tencent: Optional[TencentConfig] = Field(None, alias="tencent")
    openai_compatible: Optional[OpenAICompatibleConfig] = Field(
        None, alias="openai_compatible"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "translate_audio": Description(
            en="Enable audio translation (requires DeepLX deployment)",
            zh="启用音频翻译（需要部署 DeepLX）",
        ),
        "translate_provider": Description(
            en="Translation service provider to use", zh="要使用的翻译服务提供者"
        ),
        "deeplx": Description(
            en="Configuration for DeepLX translation service", zh="DeepLX 翻译服务配置"
        ),
        "tencent": Description(
            en="Configuration for TenCent translation service", zh="腾讯 翻译服务配置"
        ),
        "openai_compatible": Description(
            en="Configuration for OpenAI-compatible translation service",
            zh="OpenAI 兼容翻译服务配置",
        ),
    }

    @model_validator(mode="after")
    def check_translator_config(cls, values: "TranslatorConfig", info: ValidationInfo):
        translate_audio = values.translate_audio
        translate_provider = values.translate_provider

        if translate_audio:
            if translate_provider == "deeplx" and values.deeplx is None:
                raise ValueError(
                    "DeepLX configuration must be provided when translate_audio is True and translate_provider is 'deeplx'"
                )
            elif translate_provider == "tencent" and values.tencent is None:
                raise ValueError(
                    "Tencent configuration must be provided when translate_audio is True and translate_provider is 'tencent'"
                )
            elif (
                translate_provider == "openai_compatible"
                and values.openai_compatible is None
            ):
                raise ValueError(
                    "OpenAI-compatible configuration must be provided when translate_audio is True and translate_provider is 'openai_compatible'"
                )

        return values


class TTSPreprocessorConfig(I18nMixin):
    """Configuration for TTS preprocessor."""

    remove_special_char: bool = Field(..., alias="remove_special_char")
    ignore_brackets: bool = Field(default=True, alias="ignore_brackets")
    ignore_parentheses: bool = Field(default=True, alias="ignore_parentheses")
    ignore_asterisks: bool = Field(default=True, alias="ignore_asterisks")
    ignore_angle_brackets: bool = Field(default=True, alias="ignore_angle_brackets")
    translator_config: TranslatorConfig = Field(..., alias="translator_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "remove_special_char": Description(
            en="Remove special characters from the input text",
            zh="从输入文本中删除特殊字符",
        ),
        "translator_config": Description(
            en="Configuration for translation services", zh="翻译服务的配置"
        ),
    }
