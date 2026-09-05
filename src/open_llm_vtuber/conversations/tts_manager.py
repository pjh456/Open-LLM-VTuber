import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    # Number of sentences to batch into a single TTS request. Batching amortizes
    # the per-request overhead of the TTS engine and avoids playback gaps.
    TTS_BATCH_SIZE = 2
    # Max seconds to wait for the batch to fill up before flushing what we have.
    TTS_FLUSH_TIMEOUT = 1.0
    _EMPTY_TEXT_RE = re.compile(r"[\s.,!?，。！？\'\"』」）】\s]+")
    _CJK_RE = re.compile(
        r"[\u2e80-\u9fff\uf900-\ufaff\ufe30-\ufe4f\u3000-\u303f\uac00-\ud7af\uff00-\uffef]"
    )

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Sentences waiting to be batched: (tts_text, display_text, actions,
        # live2d_model, tts_engine, websocket_send, translate_engine)
        self._sentence_buffer: List[tuple] = []
        # Task that flushes the buffer if it does not fill up in time
        self._flush_task: Optional[asyncio.Task] = None

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    async def queue_sentence(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        translate_engine: Optional[Any] = None,
    ) -> None:
        """
        Queue a single sentence for TTS. Sentences are buffered and flushed as
        one TTS request once TTS_BATCH_SIZE sentences are collected, or after
        TTS_FLUSH_TIMEOUT if the stream does not keep up.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
            translate_engine: Optional translation engine applied to the batched text
        """
        if len(self._EMPTY_TEXT_RE.sub("", tts_text)) == 0:
            await self.speak(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                websocket_send=websocket_send,
            )
            return

        self._sentence_buffer.append(
            (
                tts_text,
                display_text,
                actions,
                live2d_model,
                tts_engine,
                websocket_send,
                translate_engine,
            )
        )
        if len(self._sentence_buffer) >= self.TTS_BATCH_SIZE:
            await self._flush_sentence_buffer()
        else:
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_on_timeout())

    async def _flush_on_timeout(self) -> None:
        await asyncio.sleep(self.TTS_FLUSH_TIMEOUT)
        if len(self._sentence_buffer) == 1:
            await self._flush_sentence_buffer()

    async def _flush_sentence_buffer(self) -> None:
        if not self._sentence_buffer:
            return
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        batch, self._sentence_buffer = self._sentence_buffer, []
        logger.debug(
            f"🏃 Flushing {len(batch)} batched sentence(s) to TTS: "
            f"'''{self._join_texts([item[0] for item in batch])}'''"
        )

        tts_texts = [item[0] for item in batch]
        display_texts = [item[1] for item in batch]
        actions_list = [item[2] for item in batch]
        # All sentences in one turn share the same engine/model/send references
        live2d_model, tts_engine, websocket_send, translate_engine = (
            batch[0][3],
            batch[0][4],
            batch[0][5],
            batch[0][6],
        )

        combined_tts_text = self._join_texts(tts_texts)
        if (
            translate_engine is not None
            and len(self._EMPTY_TEXT_RE.sub("", combined_tts_text)) > 0
        ):
            combined_tts_text = translate_engine.translate(combined_tts_text)
            logger.info(f"🏃 Text after translation: '''{combined_tts_text}'''...")

        await self.speak(
            tts_text=combined_tts_text,
            display_text=self._combine_display_texts(display_texts),
            actions=self._merge_actions(actions_list),
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )

    async def flush_remaining(self) -> None:
        """Flush any buffered sentence left over when the LLM stream ends."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        if self._sentence_buffer:
            await self._flush_sentence_buffer()

    @classmethod
    def _needs_space(cls, left: str, right: str) -> bool:
        if not left or not right:
            return False
        return not (
            cls._CJK_RE.search(left[-1]) is not None
            or cls._CJK_RE.search(right[0]) is not None
        )

    @classmethod
    def _join_texts(cls, texts: List[str]) -> str:
        joined = ""
        for text in texts:
            if not joined:
                joined = text
                continue
            joined += (" " if cls._needs_space(joined, text) else "") + text
        return joined

    @classmethod
    def _combine_display_texts(cls, texts: List[DisplayText]) -> DisplayText:
        first = texts[0]
        return DisplayText(
            text=cls._join_texts([t.text for t in texts]),
            name=first.name,
            avatar=first.avatar,
        )

    @staticmethod
    def _merge_actions(
        actions_list: List[Optional[Actions]],
    ) -> Optional[Actions]:
        expressions, pictures, sounds = [], [], []
        for actions in actions_list:
            if actions is None:
                continue
            if actions.expressions:
                expressions.extend(actions.expressions)
            if actions.pictures:
                pictures.extend(actions.pictures)
            if actions.sounds:
                sounds.extend(actions.sounds)
        if not (expressions or pictures or sounds):
            return None
        return Actions(
            expressions=expressions or None,
            pictures=pictures or None,
            sounds=sounds or None,
        )

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.
        """
        buffered_payloads: Dict[int, Dict] = {}

        while True:
            try:
                # Get payload from queue
                payload, sequence_number = await self._payload_queue.get()
                buffered_payloads[sequence_number] = payload

                # Send payloads in order
                while self._next_sequence_to_send in buffered_payloads:
                    next_payload = buffered_payloads.pop(self._next_sequence_to_send)
                    await websocket_send(json.dumps(next_payload))
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Queue the payload with its sequence number
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text"""
        logger.debug(f"🏃Generating audio for '''{text}'''...")
        return await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
        self._sentence_buffer.clear()
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None
