"""Voice Text Input — Astra plugin.

Голосовой набор текста в активное окно: пользователь говорит командное слово
(«напиши»), плагин печатает распознанную речь в то окно, где стоит фокус, по
мере говорения; командное слово «отправить» нажимает Enter, «отмена» стирает
набранное. Работает в любом приложении (Telegram, браузер, Word…).

The plugin declares no capabilities and no permissions: it is a fully
autonomous background process that Astra merely hosts. Everything is driven
from the Astra settings page rendered out of `[config]` in plugin.toml.
"""

import asyncio
import json
import logging
from pathlib import Path

from astra_plugin_sdk import Plugin, tool
from astra_plugin_sdk.types import SttLoadState, SttLoadStatus

from . import deps
from .dictation import DictationEngine
from .settings import ENGINE_WHISPER, Settings
from .stt_provider import SttProvider

log = logging.getLogger("voice-text-input")

STT_LANGUAGES = ["ru"]

_TRUE_WORDS = ("1", "true", "yes", "on", "да", "вкл", "включи", "включить", "enable", "start")


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_WORDS:
            return True
        if v in ("0", "false", "no", "off", "нет", "выкл", "выключи", "выключить", "disable", "stop"):
            return False
    return default


class VoiceTextInput(Plugin):
    """Hosts the dictation engine and feeds it the Astra config."""

    def __init__(self):
        super().__init__()
        self._root = Path(__file__).resolve().parent.parent
        self._engine = DictationEngine(self._root)
        self._stt = SttProvider(self._engine)
        self._config_applied = False

    @tool("Enable or disable voice dictation (voice typing into the focused window). "
          "Use when the user asks to turn voice typing on or off. "
          "Returns the new state.")
    async def tool_dictation_toggle(self, enable: bool = True) -> str:
        # Session-scoped: the toggle lives until the next config save in Astra.
        flag = _coerce_bool(enable, True)
        settings = Settings.from_config(vars(self._engine.settings))
        settings.enabled = flag
        self._engine.apply_settings(settings)
        return "Диктовка включена" if flag else "Диктовка выключена"

    @tool("Report the current state of voice dictation: enabled or disabled, "
          "the recognition engine and the live listener status. "
          "Use this to check why voice typing is not working.")
    async def tool_dictation_status(self) -> str:
        engine = self._engine
        return (f"enabled={engine.settings.enabled}; engine={engine.settings.engine}; "
                f"status={engine.status()}")

    async def on_config_changed(self, config: dict) -> None:
        """Initial config at registration and every later save."""
        try:
            settings = Settings.from_config(config)
        except Exception:
            log.exception("ignoring malformed config %r", config)
            return
        self._config_applied = True
        self._engine.apply_settings(settings)

    # ── STT provider: Astra may use this plugin as its recognizer ────────

    async def stt_get_languages(self) -> list[str]:
        return STT_LANGUAGES

    async def stt_load(self, model_path: str, use_gpu: bool) -> None:
        # The daemon's model catalog path does not apply here: models live in
        # the plugin's models/ dir. Just warm the active engine so the first
        # utterance is not slow. Self-heal deps first: Astra's runtime python
        # may miss vosk/faster-whisper (partial install) — this finishes it.
        await asyncio.to_thread(deps.ensure_core_deps)
        if self._engine.settings.engine == ENGINE_WHISPER:
            self._engine.whisper_engine().warm_up()
        else:
            self._engine.vosk_listener()

    async def stt_unload(self) -> None:
        # The Vosk model stays resident on purpose: the dictation engine shares
        # it and it is ~50 MB. Only the heavy whisper model is droppable — the
        # daemon's idle-unload then frees its RAM.
        self._engine.whisper_engine().reset()

    async def stt_load_state(self) -> SttLoadStatus:
        settings = self._engine.settings
        if settings.engine == ENGINE_WHISPER:
            whisper = self._engine.whisper_engine()
            if whisper.ready:
                return SttLoadStatus(state=SttLoadState.READY)
            if whisper.load_error:
                return SttLoadStatus(state=SttLoadState.FAILED, detail=whisper.load_error)
            return SttLoadStatus(state=SttLoadState.UNLOADED)
        state = SttLoadState.READY if self._engine.vosk_ready() else SttLoadState.UNLOADED
        return SttLoadStatus(state=state)

    async def stt_transcribe(self, audio: bytes, sample_rate: int, options=None) -> str:
        language = options.language if options is not None else ""
        return self._stt.transcribe(audio, sample_rate, language)

    async def stt_transcribe_stream(self, audio, options=None):
        async for event in self._stt.stream_transcribe(audio, options):
            yield event

    async def stt_config_fields(self) -> list[dict]:
        # Engine/model settings already live in the plugin's [config] page.
        return []

    async def health_check(self) -> tuple[bool, str]:
        # Fallback for a daemon that delivered no config at registration:
        # fetch it once, apply defaults if that fails. Keeps the engine from
        # staying off forever when `on_config_changed` never fired.
        if not self._config_applied and self.host is not None:
            self._config_applied = True
            try:
                raw = await self.host.get_config()
                config = json.loads(raw) if raw else {}
            except Exception:
                log.exception("could not fetch config; using defaults")
                config = {}
            self._engine.apply_settings(Settings.from_config(config))
        return True, self._engine.status()

    async def on_shutdown(self) -> None:
        self._engine.stop()


if __name__ == "__main__":
    VoiceTextInput().run()
