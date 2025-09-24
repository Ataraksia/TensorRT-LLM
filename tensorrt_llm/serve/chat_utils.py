<<<<<<< HEAD
import asyncio
from collections import defaultdict
=======
>>>>>>> upstream/main
from functools import partial
from typing import (Any, Callable, Coroutine, Dict, Iterable, List, Literal,
                    Optional, Tuple, TypeAlias, TypedDict, Union, cast)

<<<<<<< HEAD
from openai.types.chat import ChatCompletionContentPartImageParam
=======
from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartInputAudioParam)
>>>>>>> upstream/main
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import (ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam)
<<<<<<< HEAD
from transformers import AutoConfig, ProcessorMixin
from typing_extensions import Required

from tensorrt_llm.inputs import async_load_image, async_load_video
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
=======
from transformers import AutoConfig
from typing_extensions import Required

from tensorrt_llm.inputs import (ConversationMessage, MultimodalData,
                                 MultimodalDataTracker,
                                 add_multimodal_placeholders, async_load_audio,
                                 async_load_image, async_load_video)
>>>>>>> upstream/main
from tensorrt_llm.logger import logger


class VideoURL(TypedDict):
    """Type definition for video URL structure."""
    url: Required[str]


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Type definition for video content part parameters."""
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]


<<<<<<< HEAD
class ConversationMessage(TypedDict):
    """Type definition for conversation message structure."""
    role: str
    content: str


=======
>>>>>>> upstream/main
# Type Aliases and Constants
ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartVideoParam,
    str]

<<<<<<< HEAD
VALID_MESSAGE_CONTENT_MM_PART_TYPES = ["text", "image_url", "video_url"]
=======
# TODO: Add "input_audio" to support byte_encoded audio input.
VALID_MESSAGE_CONTENT_MM_PART_TYPES = [
    "text", "image_url", "video_url", "audio_url"
]
>>>>>>> upstream/main

# Parser Functions
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageParser = partial(cast, ChatCompletionContentPartImageParam)
_VideoParser = partial(cast, ChatCompletionContentPartVideoParam)
<<<<<<< HEAD
=======
_AudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
>>>>>>> upstream/main

MM_PARSER_MAP: dict[str, Callable[[ChatCompletionContentPartParam], Union[
    str, dict[str, str]]]] = {
        "text":
        lambda part: _TextParser(part).get("text", None),
        "image_url":
        lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
        "video_url":
        lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
<<<<<<< HEAD
    }


class AsyncMultimodalDataTracker:
    """Tracks and manages multimodal data for async processing."""

    def __init__(self, model_config: AutoConfig):
        self.model_config = model_config
        self.mm_data = defaultdict[str](list)
        self.mm_placeholder_counts = defaultdict[str](int)

    async def retrieve_all_mm_data(self) -> Optional[Dict[str, List[Any]]]:
        """Retrieve all collected multimodal data."""
        if not self.mm_data:
            return None

        return {
            modality: await asyncio.gather(*items)
            for modality, items in self.mm_data.items()
        }

    def retrieve_multimodal_placeholder(self, modality: str,
                                        current_count: int) -> Optional[str]:
        """Get the appropriate placeholder for a given modality and model type."""
        model_type = self.model_config.model_type

        if modality == "image":
            if model_type in ("qwen2_vl", "qwen2_5_vl"):
                return "<|vision_start|><|image_pad|><|vision_end|>"
            elif model_type in ("mllama", "llama4"):
                return "<|image|>"
            raise TypeError(f"Unknown {modality} model type: {model_type}")
        elif modality == "video":
            if model_type in ("qwen2_vl", "qwen2_5_vl"):
                return "<|vision_start|><|video_pad|><|vision_end|>"
            raise TypeError(f"Unknown {modality} model type: {model_type}")
        raise TypeError(f"Unknown modality: {modality}")

    def add_mm_data(self, media_type: str, data: Coroutine):
        current_count = len(self.mm_data[media_type]) + 1
        placeholder = self.retrieve_multimodal_placeholder(
            media_type, current_count)
        self.mm_data[media_type].append(data)
        if placeholder:
            self.mm_placeholder_counts[placeholder] += 1

    def mm_data_counts(self) -> Dict[str, int]:
        """Get the count of multimodal placeholders."""
        return dict(self.mm_placeholder_counts)


def add_multimodal_placeholders(text_prompt: str,
                                mm_placeholder_counts: dict[str, int]) -> str:
    """Add multimodal placeholders to the text prompt."""
    placeholders = []
    for placeholder in mm_placeholder_counts:
        placeholders.extend([placeholder] * mm_placeholder_counts[placeholder])
    return "\n".join(placeholders + [text_prompt])


=======
        "audio_url":
        lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    }


>>>>>>> upstream/main
def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam
) -> tuple[str, Union[str, dict[str, str]]]:
    """Parse a single multimodal part of a chat message."""
    assert isinstance(part, dict)
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        return part_type, MM_PARSER_MAP[part_type](part)

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


def parse_chat_message_content_part(
<<<<<<< HEAD
    part: ChatCompletionMessageParam,
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> Optional[str]:
=======
    part: ChatCompletionMessageParam, ) -> Optional[Any]:
>>>>>>> upstream/main
    """Parse a single part of a chat message."""
    if isinstance(part, str):
        return part

    part_type, content = _parse_chat_message_content_mm_part(part)

<<<<<<< HEAD
    # if part_type is text/image_url/video_url but content is None, log a warning and skip
=======
    # if part_type is text/image_url/video_url/audio_url but content is None, log a warning and skip
>>>>>>> upstream/main
    if part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') with empty / unparsable content.",
            part, part_type)
        return None

    if part_type == "text":
        return cast(str, content)

    if part_type == "image_url":
        str_content = cast(str, content)

        async def load_image_async():
            try:
                return await async_load_image(str_content)
            except Exception as e:
                logger.error(f"Failed to load image: {str(e)}")
                return None

<<<<<<< HEAD
        mm_data_tracker.add_mm_data("image", load_image_async())
        return None
=======
        return MultimodalData(modality="image", data=load_image_async())
>>>>>>> upstream/main

    if part_type == "video_url":
        str_content = cast(str, content)

        async def load_video_async():
            try:
                return await async_load_video(str_content, num_frames=8)
            except Exception as e:
                logger.error(f"Failed to load video: {str(e)}")
                return None

<<<<<<< HEAD
        mm_data_tracker.add_mm_data("video", load_video_async())
        return None
=======
        return MultimodalData(modality="video", data=load_video_async())

    if part_type == "audio_url":
        str_content = cast(str, content)

        async def load_audio_async():
            try:
                return await async_load_audio(str_content)
            except Exception as e:
                logger.error(f"Failed to load audio: {str(e)}")
                return None

        return MultimodalData(modality="audio", data=load_audio_async())
>>>>>>> upstream/main

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionMessageParam],
<<<<<<< HEAD
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> List[ConversationMessage]:
    """Parse multiple parts of a chat message."""
    content_parts = []
    for part in parts:
        parse_res = parse_chat_message_content_part(part, mm_data_tracker)
        if parse_res:
            content_parts.append(parse_res)

    text_prompt = "\n".join(content_parts)
    mm_placeholder_counts = mm_data_tracker.mm_data_counts()

    if mm_placeholder_counts:
        text_prompt = add_multimodal_placeholders(text_prompt,
                                                  mm_placeholder_counts)

    return [ConversationMessage(role=role, content=text_prompt)]


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> List[ConversationMessage]:
=======
) -> ConversationMessage:
    """Parse multiple parts of a chat message."""
    text_parts = []
    media_parts = []
    for part in parts:
        parse_res = parse_chat_message_content_part(part)
        if parse_res:
            if isinstance(parse_res, str):
                text_parts.append(parse_res)
            else:
                media_parts.append(parse_res)

    text_prompt = "\n".join(text_parts)

    return ConversationMessage(role=role,
                               content=text_prompt,
                               media=media_parts)


def parse_chat_message_content(
    message: ChatCompletionMessageParam, ) -> ConversationMessage:
>>>>>>> upstream/main
    """Parse the content of a chat message."""
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]

    result = parse_chat_message_content_parts(
        role,
        content,
<<<<<<< HEAD
        mm_data_tracker,
=======
>>>>>>> upstream/main
    )
    return result


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse multiple chat messages and return conversation and coroutine."""
    conversation = []
<<<<<<< HEAD
    mm_data_tracker = AsyncMultimodalDataTracker(model_config)

    for msg in messages:
        sub_messages = parse_chat_message_content(msg, mm_data_tracker)
        conversation.extend(sub_messages)

    return conversation, mm_data_tracker.retrieve_all_mm_data()


def resolve_hf_chat_template(
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    """Resolve the appropriate chat template to use."""

    # 1. If chat_template is not None, return it
    if chat_template is not None:
        return chat_template

    # 2. If tool is not provided, use the processor's default chat template
    if not tools and processor and hasattr(processor, 'chat_template'):
        return processor.chat_template

    # 3. If tool is provided, use the tool
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug("Failed to load AutoTokenizer chat template for %s",
                     tokenizer.name_or_path)
    return None


def apply_chat_template(
    *,
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    conversation: list[ConversationMessage],
    add_generation_prompt: bool,
    tools: Optional[list[dict[str, Any]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """Apply chat template to the conversation."""
    hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                chat_template, tools)

    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")

    return tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=hf_chat_template,
        **(chat_template_kwargs or {}),
    )
=======
    mm_placeholder_counts = []
    mm_data_tracker = MultimodalDataTracker(model_config.model_type)

    for msg in messages:
        parsed_msg = parse_chat_message_content(msg)
        conversation.append(parsed_msg)
        if parsed_msg["media"]:
            for mdata in parsed_msg["media"]:
                mm_data_tracker.add_data(mdata["modality"], mdata["data"])
        mm_placeholder_count = mm_data_tracker.placeholder_counts()
        if mm_placeholder_count:
            parsed_msg["content"] = add_multimodal_placeholders(
                model_config.model_type, parsed_msg["content"],
                mm_placeholder_count)
        mm_placeholder_counts.append(mm_placeholder_count)

    return conversation, mm_data_tracker.retrieve_all_async(
    ), mm_placeholder_counts


def check_multiple_response(n: int, backend: Optional[str]):
    if n > 1 and backend == "pytorch":
        raise ValueError(
            "Multiple response is not supported in PyTorch workflow")
>>>>>>> upstream/main
