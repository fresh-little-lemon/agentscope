# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""The main entry point of the browser agent example."""
import asyncio
import os
import sys
import argparse
import traceback
import json
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from browser_agent import BrowserAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from agentscope.mcp import StdIOStatefulClient
from agentscope.tool import Toolkit
from agentscope.mcp import StdIOStatefulClient
from agentscope.agent import UserAgent
from agentscope.message import Msg


class LoggingMemory(InMemoryMemory):
    """Memory that logs all added messages."""
    
    def __init__(self):
        super().__init__()
        self.history_log = []
        
    async def add(self, memories, marks=None, allow_duplicates=False, **kwargs):
        # Capture memories before adding to internal storage
        to_log = memories if isinstance(memories, list) else [memories]
        if to_log:
            for msg in to_log:
                if msg:
                    # Create a log entry
                    log_entry = {
                        "role": msg.role,
                        "name": msg.name,
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    }
                    if hasattr(msg, "metadata") and msg.metadata:
                        log_entry["metadata"] = msg.metadata
                    
                    self.history_log.append(log_entry)
                    
        # Call parent method
        await super().add(memories, marks, allow_duplicates, **kwargs)


class FinalResult(BaseModel):
    """A structured result model for structured output."""

    result: str = Field(
        description="The final result to the initial user query",
    )


async def main(
    start_url_param: str = "https://www.google.com",
    max_iters_param: int = 50,
    headless: bool = False,
    record_video: bool = False,
) -> None:
    """The main entry point for the browser agent example."""
    # Setup toolkit with browser tools from MCP server
    toolkit = Toolkit()
    
    # Session logging setup
    session_logs_dir = Path("session_logs")
    session_logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure browser client arguments
    # Set default viewport to 1280x720 to ensure consistent "fullscreen" experience
    client_args = ["@playwright/mcp@latest", "--viewport-size=1280x720", "--isolated"]
    
    # Prepare environment variables
    env = os.environ.copy()
    
    if headless:
        client_args.append("--headless")
        # Ensure env var is consistent if user wants headless
        env["PLAYWRIGHT_MCP_HEADLESS"] = "true"
    else:
        # User wants headed (default). Ensure env var doesn't force headless.
        # Removing it or setting to 'false'/'0' can work depending on implementation.
        # Safest is to remove it if present.
        if "PLAYWRIGHT_MCP_HEADLESS" in env:
            del env["PLAYWRIGHT_MCP_HEADLESS"]

    if record_video:
        session_video_dir = session_logs_dir / f".video_tmp_{timestamp}"
        session_video_dir.mkdir(exist_ok=True)
        client_args.append("--save-video=1280x720")
        client_args.append(f"--output-dir={session_video_dir.absolute()}")
    else:
        session_video_dir = None
        
    browser_client = StdIOStatefulClient(
        name="playwright-mcp",
        command="npx",
        args=client_args,
        env=env,
    )
    
    # Use LoggingMemory instead of list
    memory = LoggingMemory()
    
    try:
        # Connect to the browser client
        await browser_client.connect()
        await toolkit.register_mcp_client(browser_client)

        agent = BrowserAgent(
            name="Browser-Use Agent",
            model=OpenAIChatModel(
                api_key="40a114783556416892a6e3914856367f.xYQOslFt2fzVmER8",
                model_name="glm-4",
                client_kwargs={"base_url": "https://open.bigmodel.cn/api/paas/v4"},
                stream=False,
            ),
            formatter=OpenAIChatFormatter(),
            memory=memory,
            toolkit=toolkit,
            max_iters=max_iters_param,
            start_url=start_url_param,
        )
        user = UserAgent("User")

        msg = None
        while True:
            msg = await user(msg)
            
            # Log user message
            if msg:
                # Add to agent memory to ensure it's logged
                await memory.add(msg)
                
            if msg.get_text_content() == "exit":
                break
            
            # Get agent response
            msg = await agent(msg, structured_model=FinalResult)
            
            # Agent response is automatically added to memory by the agent itself

    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        print("Cleaning up browser client...")
    finally:
        # Save session history from memory
        if hasattr(memory, "history_log") and memory.history_log:
            log_file = session_logs_dir / f"session_{timestamp}.json"
            
            # Find video file if recording was enabled to include in metadata
            video_meta = {}
            if record_video and session_video_dir and session_video_dir.exists():
                webm_files = list(session_video_dir.glob("**/*.webm"))
                if webm_files:
                    latest_video = max(webm_files, key=os.path.getmtime)
                    video_meta = {
                        "original_video": str(latest_video.name),
                        "format": "webm",
                        "status": "recording_captured"
                    }

            # Prepare full log structure
            log_data = {
                "session_id": timestamp,
                "metadata": {
                    "start_url": start_url_param,
                    "max_iters": max_iters_param,
                    "headless": headless,
                    "record_video": record_video,
                    "video_info": video_meta
                },
                "history": memory.history_log
            }

            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                print(f"\nSession log saved to: {log_file}")
            except Exception as log_error:
                print(f"Error saving session log: {log_error}")

        # Ensure browser client is always closed,
        # regardless of success or failure
        try:
            # Explicitly call browser_close tool to release resources before process exit
            if browser_client is not None:
                try:
                    print("Requesting browser to close...")
                    await browser_client.call_tool("browser_close", {})
                except Exception as close_tool_err:
                    # Tool might not be available or connection already lost
                    pass

            await browser_client.close()
            print("Browser client closed successfully.")
            
            # Post-process video if recording was enabled
            if record_video and session_video_dir and session_video_dir.exists():
                # Give filesystem a moment to sync
                time.sleep(1)
                
                try:
                    # Look ONLY in the session-specific directory
                    webm_files = list(session_video_dir.glob("**/*.webm"))
                    if webm_files:
                        latest_video = max(webm_files, key=os.path.getmtime)
                        
                        # Use same timestamp as session log
                        mp4_output = session_logs_dir / f"session_{timestamp}.mp4"
                        
                        # Convert to MP4
                        if shutil.which("ffmpeg"):
                            print(f"Converting video to MP4: {mp4_output}...")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", str(latest_video),
                                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                                "-loglevel", "error",
                                str(mp4_output)
                            ], check=True)
                            print(f"Video saved to: {mp4_output}")
                            
                            # Clean up the session-specific temp directory
                            shutil.rmtree(session_video_dir)
                        else:
                            print("ffmpeg not found! Keeping original webm video.")
                            # Move webm to logs dir with correct name if no ffmpeg
                            shutil.move(str(latest_video), str(session_logs_dir / f"session_{timestamp}.webm"))
                            shutil.rmtree(session_video_dir)
                    else:
                        print(f"No video recording found in {session_video_dir}")
                except Exception as video_error:
                    print(f"Error compiling video: {video_error}")
                        
        except Exception as cleanup_error:
            print(f"Error while closing browser client: {cleanup_error}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Browser Agent Example with configurable reply method",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        default="https://www.google.com",
        help=(
            "Starting URL for the browser agent "
            "(default: https://www.google.com)"
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum number of iterations (default: 50)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: False, i.e., headed)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record session video (saved to session_logs/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting Browser Agent Example...")
    print(
        "The browser agent will use "
        "playwright-mcp (https://github.com/microsoft/playwright-mcp)."
        "Make sure the MCP server is installed "
        "by `npx @playwright/mcp@latest`",
    )
    print("\nUsage examples:")
    print("  python main.py                           # Start with defaults")
    print("  python main.py --start-url https://example.com --max-iters 100")
    print("  python main.py --headless               # Run in headless mode")
    print("  python main.py --record                 # Record session video")
    print("  python main.py --help                   # Show all options")
    print()

    # Parse command line arguments
    args = parse_arguments()

    # Get other parameters
    start_url = args.start_url
    max_iters = args.max_iters
    is_headless = args.headless
    is_record = args.record

    # Validate parameters
    if max_iters <= 0:
        print("Error: max-iters must be positive")
        sys.exit(1)

    if not start_url.startswith(("http://", "https://")):
        print("Error: start-url must be a valid HTTP/HTTPS URL")
        sys.exit(1)

    print(f"Starting URL: {start_url}")
    print(f"Maximum iterations: {max_iters}")
    print(f"Headless mode: {is_headless}")
    print(f"Record video: {is_record}")

    asyncio.run(main(start_url, max_iters, is_headless, is_record))
