import argparse
import base64
import time
import os
import sys
import threading
import shutil
import textwrap

from mlx_engine.generate import load_model, load_draft_model, create_generator, tokenize
from mlx_engine.utils.token import Token
from mlx_engine.utils.kv_cache_quantization import VALID_KV_BITS, VALID_KV_GROUP_SIZE
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from transformers import AutoTokenizer, AutoProcessor

DEFAULT_PROMPT = "Tell me about NYC"
DEFAULT_TEMP = 0.8

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="LM Studio mlx-engine inference script"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The file system path to the model",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        type=str,
        help="Message to be processed by the model. Use '-' to read from stdin",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        type=str,
        help="System prompt for the model",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Disable the system prompt",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Path of the images to process",
    )
    parser.add_argument(
        "--temp",
        default=DEFAULT_TEMP,
        type=float,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--stop-strings",
        type=str,
        nargs="+",
        help="Strings that will stop the generation",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=0,
        help="Number of top logprobs to return",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Max context size of the model",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=VALID_KV_BITS,
        help="Number of bits for KV cache quantization. Must be between 3 and 8 (inclusive)",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        choices=VALID_KV_GROUP_SIZE,
        help="Group size for KV cache quantization",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        help="When --kv-bits is set, start quantizing the KV cache from this step onwards",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="The file system path to the draft model for speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
    )
    parser.add_argument(
        "--print-prompt-progress",
        action="store_true",
        help="Enable printed prompt processing progress callback",
    )
    parser.add_argument(
        "--max-img-size", type=int, help="Downscale images to this side length (px)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent generation threads to run (default: 1)"
    )
    return parser


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None

    def add_tokens(self, tokens: list[Token]):
        """Record new tokens and their timing."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        draft_tokens = sum(1 for token in tokens if token.from_draft)
        if self.num_accepted_draft_tokens is None:
            self.num_accepted_draft_tokens = 0
        self.num_accepted_draft_tokens += draft_tokens

        self.total_tokens += len(tokens)

    def print_stats(self):
        """Print generation statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time
        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        print("\n\nGeneration stats:")
        print(f" - Tokens per second: {tokens_per_second:.2f}")
        if self.num_accepted_draft_tokens is not None:
            print(
                f" - Number of accepted draft tokens: {self.num_accepted_draft_tokens}"
            )
        print(f" - Time to first token: {time_to_first_token:.2f}s")
        print(f" - Total tokens generated: {self.total_tokens}")
        print(f" - Total time: {total_time:.2f}s")


def resolve_model_path(model_arg):
    # If it's a full path or local file, return as-is
    if os.path.exists(model_arg):
        return model_arg

    # Check common local directories
    local_paths = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]

    for path in local_paths:
        full_path = os.path.join(path, model_arg)
        if os.path.exists(full_path):
            return full_path

    raise ValueError(f"Could not find model '{model_arg}' in local directories")


# Global lock for printing to avoid interleaving
print_lock = threading.Lock()


class ColumnDisplay:
    """Manages side-by-side column display for concurrent generation threads."""

    def __init__(self, num_columns=2):
        self.num_columns = num_columns
        self.terminal_width = shutil.get_terminal_size().columns

        # Reserve space for separators between columns
        separator_space = num_columns - 1
        self.column_width = (self.terminal_width - separator_space) // num_columns

        # Ensure minimum column width
        if self.column_width < 40:
            print(f"Warning: Terminal width ({self.terminal_width}) is too narrow for {num_columns} columns.")
            print(f"Each column will be {self.column_width} characters wide.")

        self.buffers = {i: "" for i in range(1, num_columns + 1)}
        self.completed = {i: False for i in range(1, num_columns + 1)}
        self.lock = threading.Lock()

        # Clear screen and hide cursor
        print("\033[2J\033[H", end="", flush=True)

    def append_text(self, thread_id, text):
        """Append text to a thread's buffer and redraw."""
        with self.lock:
            self.buffers[thread_id] += text
            self._redraw()

    def mark_complete(self, thread_id, stats_text):
        """Mark a thread as complete with stats."""
        with self.lock:
            self.completed[thread_id] = True
            self.buffers[thread_id] += f"\n\n{stats_text}"
            self._redraw()

    def _wrap_text(self, text, width):
        """Wrap text to fit within column width, preserving intentional breaks."""
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append('')
            else:
                wrapped = textwrap.fill(
                    paragraph,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False
                )
                lines.extend(wrapped.split('\n'))
        return lines

    def _redraw(self):
        """Redraw all columns."""
        # Move cursor to top
        print("\033[H", end="", flush=True)

        # Split each buffer into wrapped lines
        wrapped_columns = []
        max_lines = 0

        for thread_id in range(1, self.num_columns + 1):
            header = f"{'='*5} Thread {thread_id} {'='*5}"
            content_lines = self._wrap_text(self.buffers[thread_id], self.column_width)
            lines = [header, ""] + content_lines
            wrapped_columns.append(lines)
            max_lines = max(max_lines, len(lines))

        # Print rows with columns side by side
        for row_idx in range(max_lines):
            row_parts = []
            for col_idx in range(self.num_columns):
                lines = wrapped_columns[col_idx]
                if row_idx < len(lines):
                    text = lines[row_idx]
                    # Truncate and pad to column width
                    text = text[:self.column_width].ljust(self.column_width)
                else:
                    text = " " * self.column_width
                row_parts.append(text)

            print("|".join(row_parts))

        # Clear to end of screen
        print("\033[J", end="", flush=True)


display = None


def run_generation_thread(
    thread_id,
    model_kit,
    prompt_tokens,
    images_b64,
    max_img_size,
    stop_strings,
    max_tokens,
    top_logprobs,
    prompt_progress_reporter,
    num_draft_tokens,
    temp,
):
    """Run a single generation stream in a thread."""
    global display
    stats_collector = GenerationStatsCollector()
    logprobs_list = []

    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_b64,
        max_image_size=max_img_size,
        stop_strings=stop_strings,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
        prompt_progress_reporter=prompt_progress_reporter,
        num_draft_tokens=num_draft_tokens,
        temp=temp,
    )

    for generation_result in generator:
        display.append_text(thread_id, generation_result.text)
        stats_collector.add_tokens(generation_result.tokens)
        logprobs_list.extend(generation_result.top_logprobs)

        if generation_result.stop_condition:
            # Build stats text
            end_time = time.time()
            total_time = end_time - stats_collector.start_time
            time_to_first_token = stats_collector.first_token_time - stats_collector.start_time if stats_collector.first_token_time else 0
            effective_time = total_time - time_to_first_token
            tokens_per_second = (
                stats_collector.total_tokens / effective_time if effective_time > 0 else float("inf")
            )

            stats_text = f"COMPLETE\n"
            stats_text += f"Tokens/sec: {tokens_per_second:.2f}\n"
            stats_text += f"Total tokens: {stats_collector.total_tokens}\n"
            stats_text += f"Stop: {generation_result.stop_condition.stop_reason}"

            display.mark_complete(thread_id, stats_text)


if __name__ == "__main__":
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    if isinstance(args.images, str):
        args.images = [args.images]

    # Load the model
    model_path = resolve_model_path(args.model)
    print("Loading model...", end="\n", flush=True)
    model_kit = load_model(
        str(model_path),
        max_kv_size=args.max_kv_size,
        trust_remote_code=False,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    print("\rModel load complete ✓", end="\n", flush=True)

    # Load draft model if requested
    if args.draft_model:
        load_draft_model(model_kit=model_kit, path=resolve_model_path(args.draft_model))

    # Tokenize the prompt
    prompt = args.prompt
    if prompt == "-":
        stdin_prompt = sys.stdin.read()
        prompt = stdin_prompt

    # Build conversation with optional system prompt
    conversation = []
    if not args.no_system:
        conversation.append({"role": "system", "content": args.system})

    # Handle the prompt according to the input type
    # If images are provided, add them to the prompt
    images_base64 = []
    if args.images:
        tf_tokenizer = AutoProcessor.from_pretrained(model_path)
        images_base64 = [image_to_base64(img_path) for img_path in args.images]
        conversation.append(
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "base64": image_b64}
                        for image_b64 in images_base64
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        )
    else:
        tf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        conversation.append({"role": "user", "content": prompt})
    prompt = tf_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenize(model_kit, prompt)

    # Clamp image size
    max_img_size = (args.max_img_size, args.max_img_size) if args.max_img_size else None

    # Prepare prompt progress reporter
    prompt_progress_reporter = LoggerReporter() if args.print_prompt_progress else None

    # Initialize column display
    display = ColumnDisplay(num_columns=args.parallel)

    # Create and start all threads
    threads = []
    for thread_id in range(1, args.parallel + 1):
        thread = threading.Thread(
            target=run_generation_thread,
            args=(
                thread_id,
                model_kit,
                prompt_tokens,
                images_base64,
                max_img_size,
                args.stop_strings,
                1024,
                args.top_logprobs,
                prompt_progress_reporter,
                args.num_draft_tokens,
                args.temp,
            ),
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Move cursor below the display
    print("\n" * 3)
    print("=== All generation threads completed ===")
