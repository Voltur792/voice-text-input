"""Plugin settings: defaults and tolerant parsing of the Astra config dict.

The daemon delivers whatever the user saved (or an older schema, or `{}` on a
fresh install), so every field is coerced defensively and never raises.
"""

from dataclasses import dataclass

ENGINE_WHISPER = "faster-whisper"
ENGINE_VOSK = "vosk"
ENGINE_YANDEX = "yandex"
ENGINE_OPENAI = "openai"
ENGINE_GOOGLE = "google"
ENGINES = (ENGINE_WHISPER, ENGINE_VOSK, ENGINE_YANDEX, ENGINE_OPENAI, ENGINE_GOOGLE)

#: Engines that refine finished segments over the network (Vosk still types
#: the live draft); each needs its key, otherwise the code falls back to vosk.
CLOUD_ENGINES = (ENGINE_YANDEX, ENGINE_OPENAI, ENGINE_GOOGLE)

WHISPER_MODELS = ("tiny", "base", "small")
SEND_KEYS = ("enter", "ctrl+enter")

_TRUE = ("1", "true", "yes", "on", "да", "вкл")
_FALSE = ("0", "false", "no", "off", "нет", "выкл")


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return default


def _as_text(value, default: str) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v:
            return v
    return default


def _as_choice(value, choices, default: str) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in choices:
            return v
    return default


def _as_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


@dataclass
class Settings:
    """Everything the dictation engine needs, with working defaults."""

    enabled: bool = True
    start_word: str = "напиши"
    send_word: str = "отправить"
    finish_word: str = "закончить"
    cancel_word: str = "отмена"
    new_line_word: str = "с новой строки"
    wake_words: str = "астра"
    engine: str = ENGINE_WHISPER
    whisper_model: str = "small"
    yandex_api_key: str = ""
    yandex_lang: str = "ru-RU"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "whisper-1"
    google_api_key: str = ""
    send_key: str = "enter"
    corrections: bool = True
    language: str = "ru"
    send_wait_secs: float = 1.2

    @property
    def wake_word_list(self) -> list[str]:
        """Parsed wake phrases; empty strings dropped."""
        return [p.strip().lower() for p in (self.wake_words or "").split(",") if p.strip()]

    @property
    def yandex_configured(self) -> bool:
        return bool(self.yandex_api_key.strip())

    @classmethod
    def from_config(cls, config) -> "Settings":
        if not isinstance(config, dict):
            config = {}
        return cls(
            enabled=_as_bool(config.get("enabled"), True),
            start_word=_as_text(config.get("start_word"), "напиши"),
            send_word=_as_text(config.get("send_word"), "отправить"),
            # empty means "feature off" for these two, so keep "" as-is
            finish_word=(config.get("finish_word") if isinstance(config.get("finish_word"), str)
                         else "закончить"),
            cancel_word=_as_text(config.get("cancel_word"), "отмена"),
            new_line_word=(config.get("new_line_word") if isinstance(config.get("new_line_word"), str)
                           else "с новой строки"),
            wake_words=(config.get("wake_words") if isinstance(config.get("wake_words"), str)
                        else "астра"),
            engine=_as_choice(config.get("engine"), ENGINES, ENGINE_WHISPER),
            whisper_model=_as_choice(config.get("whisper_model"), WHISPER_MODELS, "small"),
            yandex_api_key=(config.get("yandex_api_key").strip()
                            if isinstance(config.get("yandex_api_key"), str) else ""),
            yandex_lang=_as_text(config.get("yandex_lang"), "ru-RU")[:12],
            openai_api_key=(config.get("openai_api_key").strip()
                            if isinstance(config.get("openai_api_key"), str) else ""),
            openai_base_url=((config.get("openai_base_url").strip().rstrip("/")
                              if isinstance(config.get("openai_base_url"), str) else "")
                             or "https://api.openai.com/v1"),
            openai_model=(config.get("openai_model").strip()
                          if isinstance(config.get("openai_model"), str) else "") or "whisper-1",
            google_api_key=(config.get("google_api_key").strip()
                            if isinstance(config.get("google_api_key"), str) else ""),
            send_key=_as_choice(config.get("send_key"), SEND_KEYS, "enter"),
            corrections=_as_bool(config.get("corrections"), True),
            language=_as_text(config.get("language"), "ru")[:5].lower() or "ru",
            send_wait_secs=_as_float(config.get("send_wait_secs"), 1.2, 0.0, 5.0),
        )
