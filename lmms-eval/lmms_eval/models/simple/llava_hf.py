import warnings
from typing import List, Optional, Tuple, Union
import os
import re
import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.kv_carryover import (
    ChatAdapter,
    carryover_enabled,
    collapse_placeholders,
    run_conversation,
    score_references_carryover,
)
from lmms_eval.models.model_utils.reference_scoring import (
    ReferenceScores,
    forced_decode_nll,
    ppl_on_model_history,
    reference_answers,
    score_references,
)

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

import types
from utils.model_llama_patches import patch_llama
from utils.model_qwen2_patches import patch_qwen2
from utils.generate_patches import generate

patch_llama()
patch_qwen2()

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Default chat for llava-hf/llava-1.5 models: https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

model_map = {
    "llava": LlavaForConditionalGeneration,
    "llava_next": LlavaNextForConditionalGeneration,
}

try:
    from transformers import LlavaOnevisionForConditionalGeneration

    model_map["llava_onevision"] = LlavaOnevisionForConditionalGeneration
except Exception as e:
    eval_logger.debug("Transformers version does not support llava-onevision. Skipping.")


@register_model("llava_hf")
class LlavaHf(lmms):
    """
    Llava Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the InstructBLIP model in lmms_eval/models/instructblip.py

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        config = AutoConfig.from_pretrained(pretrained)
        self.max_frames_num = max_frames_num
        model_type = getattr(config, "model_type", "llava")
        model_type = model_map[model_type]
        model_kwargs = {
            "revision": revision,
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_implementation,
        }
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        self._model = model_type.from_pretrained(pretrained, **model_kwargs)
        self._model.generate = types.MethodType(generate, self._model)

        self.pretrained = pretrained
        if getattr(config, "model_type", "llava") == "llava_onevision":
            vision_aspect_ratio = os.environ.get("VISION_ASPECT_RATIO", "anyres_max_9")
            self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code, vision_aspect_ratio=vision_aspect_ratio)
            self._model.config.vision_aspect_ratio = vision_aspect_ratio
        else:
            self._image_processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        # Pad from left for batched generation: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._image_processor.tokenizer.padding_side = "left"
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
            image_tokens = " ".join(image_tokens)
            context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}, {"role": "assistant", "content": continuation}]
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            elif self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_contexts = [prompt]
            formatted_continuation = [prompt_and_continuation]
            model_inputs = self._image_processor(text=formatted_continuation, images=visuals, return_tensors="pt").to(self._device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            contxt_id = self._image_processor(text=formatted_contexts, return_tensors="pt")["input_ids"]
            labels[:, : contxt_id.shape[1]] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id}:\n\n{formatted_contexts[0]}\n")
                eval_logger.debug(f"Prompt and continuation for doc ID {doc_id}:\n\n{formatted_continuation[0]}\n")

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            elif isinstance(visuals[0], str):
                task_type = "video"
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                if task_type == "image":
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                elif task_type == "video":
                    image_tokens = [DEFAULT_VIDEO_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}]
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            if task_type == "video":
                try:
                    visuals = [self.load_video(visuals, self.max_frames_num)]
                except Exception as e:
                    res.append("")
                    eval_logger.info(f"Error {e} when loading video : {visuals}")
                    pbar.update(1)

            if task_type == "image":
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)
            elif task_type == "video":
                inputs = self._image_processor(videos=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = True if gen_kwargs["temperature"] > 0 else False
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1] :]
            except Exception:
                eval_logger.exception("Error in generating")
                cont = inputs["input_ids"].new_empty((inputs["input_ids"].shape[0], 0))
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[Tuple[str, ...]]:
        """Multi-round generation that keeps a real chat history.

        Round 0 sends the task context with the doc's visuals. After each round the task's
        `doc_to_text(doc, previous_output=..., round_idx=...)` returns
        `(visuals, context, terminal, previous_output, round_info)`:
          * a list `context` of {"role", "content"} messages is the full chat history; the round-0
            visuals are attached to its first user message (e.g. ConvBench);
          * a string `context` is sent as a fresh single-turn prompt with the returned visuals.
        By default (MULTI_TURN_KV_CARRYOVER=1) the KV cache is carried from one turn to the next: later turns
        prefill only their new tokens on top of the (TGV-KV compressed) cache, see
        `lmms_eval.models.model_utils.kv_carryover`. With MULTI_TURN_KV_CARRYOVER=0 every turn is a separate
        `generate` call over the full history.
        Returns one tuple of per-round answers per request. If the task's `doc_to_text` accepts
        `reference_round`, the tuple ends with a dict of reference NLLs for perplexity
        (see `lmms_eval.models.model_utils.reference_scoring`).
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            doc_visuals = self.flatten([doc_to_visual(doc)])
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if k != "until"}
            max_tokens = gen_kwargs.get("max_new_tokens", 1024)
            if carryover_enabled():
                adapter = self._carryover_adapter(doc_visuals, gen_kwargs)
                if ppl_on_model_history():
                    outputs, reference_scores = run_conversation(self.model, adapter, doc, context, doc_to_text, max_tokens, references=reference_answers(doc_to_text, doc))
                else:
                    outputs, _ = run_conversation(self.model, adapter, doc, context, doc_to_text, max_tokens)
                    reference_scores = score_references_carryover(self.model, adapter, doc, doc_to_text, max_tokens)
            else:
                outputs, reference_scores = self._multi_round_reencode(doc, context, doc_visuals, doc_to_text, gen_kwargs)
            result = tuple(outputs) + ((reference_scores,) if reference_scores else ())
            res.append(result)
            self.cache_hook.add_partial("generate_until_multi_round", (request.args[0], gen_kwargs), result)
            pbar.update(1)
        pbar.close()
        return res

    def _multi_round_reencode(self, doc, context, doc_visuals, doc_to_text, gen_kwargs):
        """One `generate` call per turn over the full history (MULTI_TURN_KV_CARRYOVER=0)."""
        visuals = doc_visuals
        messages = [{"role": "user", "content": context}]
        outputs, round_info = [], None
        tokenize = lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
        max_tokens = gen_kwargs.get("max_new_tokens", 1024)
        own = ppl_on_model_history()
        scores = ReferenceScores(reference_answers(doc_to_text, doc) if own else [], tokenize, max_tokens)
        while True:
            # PPL_HISTORY=model: reference k after the same history answer k is generated from
            scores.add(len(outputs) + 1, lambda ids, m=messages, v=visuals: forced_decode_nll(self.model, self._chat_inputs(m, v), ids, self.eot_token_id, self.eot_token_id))
            outputs.append(self._generate_chat(messages, visuals, gen_kwargs))
            new_visuals, context, terminal, _, round_info = doc_to_text(doc, previous_output=list(outputs), round_idx=len(outputs), previous_round_info=round_info)
            if terminal:
                break
            if isinstance(context, list):
                messages = context
            else:
                messages = [{"role": "user", "content": context}]
                visuals = self.flatten([new_visuals]) if new_visuals else visuals
        if own:
            return outputs, scores.result()
        reference_scores = score_references(
            doc,
            doc_to_text,
            tokenize=tokenize,
            score=lambda msgs, ids: forced_decode_nll(self.model, self._chat_inputs(msgs, doc_visuals), ids, self.eot_token_id, self.eot_token_id),
            max_tokens=max_tokens,
        )
        return outputs, reference_scores

    def _carryover_adapter(self, visuals, gen_kwargs):
        """What the KV carry-over conversation loop needs from LLaVA."""
        temperature = gen_kwargs.get("temperature", 0) or 0
        do_sample = temperature > 0
        return ChatAdapter(
            chat_text=lambda messages: self._chat_text(messages, visuals),
            first_inputs=lambda messages: self._chat_inputs(messages, visuals),
            tokenize=lambda text: self.tokenizer(text)["input_ids"],
            tokenize_answer=lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"],
            decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            decode_prompt=lambda ids: collapse_placeholders(self.tokenizer.decode(ids), DEFAULT_IMAGE_TOKEN),
            generate_kwargs=dict(
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=gen_kwargs.get("top_p", None),
                num_beams=gen_kwargs.get("num_beams", 1),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                use_cache=self.use_cache,
                pad_token_id=self.eot_token_id,
                eos_token_id=self.eot_token_id,
            ),
            eos_ids={self.eot_token_id},
            pad_id=self.eot_token_id,
            image_token_ids={self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)},
        )

    def _chat_text(self, messages, visuals):
        """Prompt text for a chat history ending in a user turn (image placeholders not yet expanded).

        Checkpoints whose processor ships a chat template (e.g. llava-hf/llava-1.5-*-hf) get structured
        {"type": "image"/"text"} content, as lmms-eval's chat-mode LlavaHf renders them; such templates
        silently drop plain-string content. Other checkpoints use the plain-string (Vicuna) path.
        """
        if self.chat_template is None and getattr(self._image_processor, "chat_template", None):
            text = self._image_processor.apply_chat_template(self._structured_messages(messages, visuals), tokenize=False, add_generation_prompt=True)
        else:
            text = self._string_chat_prompt(messages, visuals)
        question = messages[-1]["content"].replace(DEFAULT_IMAGE_TOKEN, "").strip()[:50]
        if question and question not in text:
            raise ValueError(f"The chat template dropped the user message; rendered prompt: {text[:300]!r}")
        return text

    def _chat_inputs(self, messages, visuals):
        """Model inputs for a chat history ending in a user turn, with `visuals` attached to the first user message."""
        text = self._chat_text(messages, visuals)
        if not visuals:
            return self.tokenizer(text, return_tensors="pt").to(self._device)
        if isinstance(visuals[0], str):
            return self._image_processor(videos=[self.load_video(visuals, self.max_frames_num)], text=text, return_tensors="pt").to(self._device, self.model.dtype)
        return self._image_processor(images=visuals, text=text, return_tensors="pt").to(self._device, self.model.dtype)

    @staticmethod
    def _structured_messages(messages, visuals):
        """Chat messages with typed content parts; the visuals go in the first user message, before its text."""
        media = [{"type": "video"}] if visuals and isinstance(visuals[0], str) else [{"type": "image"}] * len(visuals)
        structured, attached = [], False
        for m in messages:
            content = [{"type": "text", "text": m["content"].replace(DEFAULT_IMAGE_TOKEN, "")}]
            if m["role"] == "user" and not attached:
                content, attached = media + content, True
            structured.append({"role": m["role"], "content": content})
        return structured

    def _string_chat_prompt(self, messages, visuals):
        """Prompt from plain-string messages, with image/video tokens prepended to the first user message."""
        messages = [dict(m) for m in messages]
        if visuals and not any(DEFAULT_IMAGE_TOKEN in m["content"] or DEFAULT_VIDEO_TOKEN in m["content"] for m in messages):
            is_video = isinstance(visuals[0], str)
            token = DEFAULT_VIDEO_TOKEN if is_video else DEFAULT_IMAGE_TOKEN
            first_user = next(m for m in messages if m["role"] == "user")
            first_user["content"] = f"{' '.join([token] * (1 if is_video else len(visuals)))}\n{first_user['content']}"

        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template
        elif self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_chat(self, messages, visuals, gen_kwargs) -> str:
        """Generate one assistant reply for a chat history, attaching `visuals` to the first user message.

        Errors are raised, not swallowed: a systematic failure (e.g. a prompt/template mismatch) would otherwise
        score every conversation on empty answers.
        """
        inputs = self._chat_inputs(messages, visuals)




        if os.environ.get("CONVBENCH_DEBUG_PROMPTS"):
            ids = inputs["input_ids"][0]
            image_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            n_image = int((ids == image_id).sum())
            turn = sum(m["role"] == "user" for m in messages)
            prompt = re.sub(r"(?:<image>\s*)+", f"<image x{n_image}> ", self.tokenizer.decode(ids, skip_special_tokens=False))
            eval_logger.info(f"[turn {turn}] prompt tokens: {ids.numel()} = {n_image} image + {ids.numel() - n_image} text\n{prompt}")


        temperature = gen_kwargs.get("temperature", 0) or 0
        do_sample = temperature > 0
        cont = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=gen_kwargs.get("top_p", None),
            num_beams=gen_kwargs.get("num_beams", 1),
            max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
            use_cache=self.use_cache,
            pad_token_id=self.eot_token_id,
            eos_token_id=self.eot_token_id,
        )
        cont = cont[:, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
