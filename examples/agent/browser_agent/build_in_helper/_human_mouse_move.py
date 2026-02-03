# -*- coding: utf-8 -*-
"""Human-like mouse movement skill for the browser agent."""

from __future__ import annotations
from typing import Any, Optional
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

async def human_mouse_move(
    browser_agent: Any,
    ref: str = None,
    steps: int = 500,
    moves: int = 5,
    delay_ms: int = 5000,
) -> ToolResponse:
    """
    Simulate human-like mouse movement within a specific element or across the viewport.
    This tool performs multiple randomized, smooth mouse movements.

    Args:
        ref (str): The snapshot reference (e.g., "[ref=e12]") of the element to move within. 
                             If not provided, moves across the entire viewport.
        steps (int): Number of intermediate steps for each movement. Default is 500. 
                     Increase this for slower, more continuous movement.
        moves (int): Number of randomized moves to make. Default is 5.
        delay_ms (int): Pause duration (in milliseconds) between randomized moves. 
                        Default is 5000ms. Increase to make the overall sequence slower.

    Returns:
        ToolResponse: Status of the mouse movement.
    """
    
    # Construct the JavaScript for Playwright page.evaluate / browser_run_code
    # We use browser_run_code if available via the toolkit, or construct a script.
    
    js_code = f"""
    async (page) => {{
        const getRandomInRange = (min, max) => Math.random() * (max - min) + min;
        
        let targetX, targetY, width, height;
        
        if ("{ref}" && "{ref}" !== "None") {{
            const element = await page.$( "{ref}" );
            if (element) {{
                const box = await element.boundingBox();
                if (box) {{
                    targetX = box.x;
                    targetY = box.y;
                    width = box.width;
                    height = box.height;
                }}
            }}
        }}
        
        if (!width) {{
            const viewport = page.viewportSize() || {{ width: 1280, height: 720 }};
            targetX = 0;
            targetY = 0;
            width = viewport.width;
            height = viewport.height;
        }}

        for (let i = 0; i < {moves}; i++) {{
            const x = getRandomInRange(targetX, targetX + width);
            const y = getRandomInRange(targetY, targetY + height);
            
            await page.mouse.move(x, y, {{ steps: {steps} }});
            // Use page.waitForTimeout which is available in Playwright context
            // Add a bit of jitter to the user-provided delay
            const pause = {delay_ms} * getRandomInRange(0.8, 1.2);
            await page.waitForTimeout(pause);
        }}
        
        return `Performed {moves} smooth mouse movements (steps={steps}, delay={delay_ms}ms) within area: ${{(width).toFixed(0)}}x${{(height).toFixed(0)}}`;
    }}
    """

    try:
        # Call the browser_run_code tool registered in the browser_agent's toolkit
        tool_call = {
            "name": "browser_run_code",
            "arguments": {"code": js_code}
        }
        
        # We need to execute this tool call via the browser_agent
        # browser_agent.toolkit is a Toolkit object.
        # However, it involves id generation and result handling.
        # A simpler way is to use the existing browser_agent._acting flow or call the toolkit directly.
        
        # In agentscope's browser_agent, we can use call_tool_function if we have a ToolUseBlock
        import uuid
        from agentscope.message import ToolUseBlock
        
        mcp_tool_call = ToolUseBlock(
            id=str(uuid.uuid4()),
            name="browser_run_code",
            input={"code": js_code},
            type="tool_use"
        )
        
        response = await browser_agent.toolkit.call_tool_function(mcp_tool_call)
        
        result_text = ""
        async for chunk in response:
            if chunk.content and "text" in chunk.content[0]:
                result_text = chunk.content[0]["text"]
        
        return ToolResponse(
            content=[TextBlock(type="text", text=result_text or "Mouse movement command executed.")],
            metadata={"success": True}
        )
        
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"Error during human-like mouse movement: {str(e)}")],
            metadata={"success": False}
        )
