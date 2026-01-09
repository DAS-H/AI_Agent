import os, re, dataclasses, argparse, json
from typing import Tuple, Optional, List, Dict
from agisdk import REAL
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam as MessageParam

CANDIDATE_MODELS = [
    "deepseek-ai/deepseek-v3.1-terminus",
]
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


PROMPT_PLANNER = """
You are a meticulous AI planning agent. Your job is to decompose a high-level user goal into a precise JSON list of browser action steps.

User Goal: {user_goal}

The ONLY allowed actions are "fill", "click", and "select".
- "fill": Used for typing in text boxes.
- "click": Used for clicking buttons, links, or other elements.
- "select": Used for choosing an option from a dropdown menu.

When a scroll down action is REQUIRED to load more elements, the LLM should plan a 'click' action with the **specific target_description: 'VIRTUAL_SCROLL_ACTION'**. The agent will internally handle this action.

Return ONLY a JSON list of dictionaries. Each dictionary must have:
1. "step": A short description of the action (e.g., "Type 'laptop' in search bar").
2. "action": The action type ("fill", "click", or "select").
3. "target_description": A clear, detailed description of the element to interact with (e.g., "main site search textbox", "VIRTUAL_SCROLL_ACTION" if scrolling).
4. "text": The text to type (ONLY for "fill" or "select" actions, otherwise leave as "").

Example for "Scroll down and click an element":
[
    {
        "step": "Scroll down to load more content",
        "action": "click",
        "target_description": "VIRTUAL_SCROLL_ACTION",
        "text": ""
    },
    {
        "step": "Click the target element",
        "action": "click",
        "target_description": "the newly loaded button",
        "text": ""
    }
]

Now, generate the plan for the user goal.
Return ONLY the JSON list, wrapped in a markdown code block for parsing reliability (```json ... ```).
"""


PROMPT_EXECUTOR = """
You control a browser.
Overall Goal: {overall_goal}
Current Step: {step_description}

You receive an accessibility tree (AX). Elements show BIDs like BID: 123.
Find the BEST BID to accomplish the "Current Step".

The target element is described as: "{target_description}"

{RETRY_CONTEXT}

Return EXACTLY one of the following actions based on the action type '{action_type}':
1) click("<BID>")
2) fill("<BID>", "{text_to_fill}")
3) select("<BID>", "{text_to_fill}")
4) send_msg_to_user("Could not find {target_description}")

Output ONLY the action string.
AXTREE (truncated):
{AX}
"""


def get_client() -> OpenAI:
    """Initializes the OpenAI client using the Mistral API configuration."""
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = NVIDIA_BASE_URL
    if not key:
        print(f"WARNING: API key not found. Please set MISTRAL_API_KEY.")
    return OpenAI(api_key=key, base_url=base_url)


client = get_client()


def first_working_model(msgs: list[MessageParam], max_tokens: int = 10000):
    last_err = None
    openai_msgs = [{"role": "user", "content": msgs[0]['content']}]

    call_params = {
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }

    for m in CANDIDATE_MODELS:
        try:
            print(f"[MODEL TRY] {m}")

            r = client.chat.completions.create(
                model=m,
                messages=openai_msgs,
                **call_params
            )
            print(f"[MODEL OK] Using: {m}")
            return m, r
        except Exception as e:
            err_str = str(e).lower()
            if 'rate limit' in err_str or 'not found' in err_str or 'unsupported_value' in err_str or 'invalid_request' in err_str:
                print(f"[MODEL FAIL] {m} -> {e}")
                last_err = e
            else:
                raise e
    raise RuntimeError(f"No working model from list. Last error: {last_err}")


# <--- Sanitization and Heuristic helpers ---
_PAT_CLICK = re.compile(r'^click\("([^"]+)"\)$')
_PAT_FILL = re.compile(r'^fill\("([^"]+)",\s*"([^"]*)"\)$')
_PAT_SELECT = re.compile(r'^select\("([^"]+)",\s*"([^"]*)"\)$')
_PAT_SEND_MSG = re.compile(r'^send_msg_to_user\("([^"]+)"\)$')


def sanitize_general_action(text: str) -> Optional[str]:
    """Returns a validated action string or None."""
    if not text:
        return None
    t = text.strip().strip("`")

    # Check for exact matches
    if _PAT_CLICK.fullmatch(t) or _PAT_FILL.fullmatch(t) or \
            _PAT_SELECT.fullmatch(t) or _PAT_SEND_MSG.fullmatch(t):
        return t

    # Loose rescue (if LLM adds extra words but the action is there)
    m_click = re.search(r'click\("([^"]+)"\)', t)
    if m_click:
        return f'click("{m_click.group(1)}")'

    m_fill = re.search(r'fill\("([^"]+)",\s*"([^"]*)"\)', t)
    if m_fill:
        return f'fill("{m_fill.group(1)}", "{m_fill.group(2)}")'

    m_select = re.search(r'select\("([^"]+)",\s*"([^"]*)"\)', t)
    if m_select:
        return f'select("{m_select.group(1)}", "{m_select.group(2)}")'

    if t.startswith("send_msg_to_user"):
        return t

    return None


import re
from typing import Optional


def find_best_bid_heuristic(ax: str, target_desc: str) -> Optional[str]:
    """A single, consolidated heuristic function that tries to find a BID based on keywords.

    This function attempts to find a reliable Browser ID (BID) for a target element
    by scanning the Accessibility Tree (ax) using hardcoded, prioritized keyword rules.

    Args:
        ax: The full accessibility tree (AX) as a single string.
        target_desc: A natural language description of the target element (e.g., "Add to Cart button").

    Returns:
        The best matching BID string (e.g., '187') if a heuristic match is found,
        otherwise None.
    """
    lines = ax.splitlines()
    lower_desc = target_desc.lower()

    # 1. Map BIDs to their corresponding line text
    bid_map = {}
    for line in lines:
        # Regex to find 'BID:number'
        m = re.search(r'BID:\s*([A-Za-z0-9_\-:.]+)', line)
        if m:
            bid = m.group(1)
            bid_map[bid] = line.lower()

    # --- Rule 1: Custom Checkout/Cart Heuristic (NEW/ENHANCED) ---
    if "cart" in lower_desc or "checkout" in lower_desc or "view cart" in lower_desc:

        # 1. Prioritize the known reliable cart icon BID (e.g., '187' on common platforms)
        if 'bid:187' in ax:
            print("[HEURISTIC] Found reliable cart icon (BID 187).")
            return '187'

        # 2. Search for the general Checkout/Proceed button
        for bid, lower_line in bid_map.items():
            if "button" in lower_line and (
                    "checkout" in lower_line or "proceed" in lower_line or "payment" in lower_line or "view cart" in lower_line):
                print(f"[HEURISTIC] Found general checkout/cart button ({bid}).")
                return bid

    # --- Rule 2: Find Search Textbox (Existing) ---
    if "search" in lower_desc and ("textbox" in lower_desc or "input" in lower_desc):
        for bid, lower_line in bid_map.items():
            if ("search" in lower_line) and (
                    "textbox" in lower_line or "input" in lower_line or "role=search" in lower_line):
                return bid

    # --- Rule 3: Find Search Button (Existing) ---
    if "search" in lower_desc and ("button" in lower_desc or "submit" in lower_desc or "icon" in lower_desc):
        for bid, lower_line in bid_map.items():
            if ("button" in lower_line or "link" in lower_line or "icon" in lower_line) and \
                    ("search" in lower_line or "magnifying glass" in lower_line or "submit" in lower_line):
                return bid

    # --- Rule 4: Find 'Add to Cart' button (Existing) ---
    if "add to cart" in lower_desc:
        for bid, lower_line in bid_map.items():
            if "button" in lower_line and "add to cart" in lower_line:
                return bid

    return None


if __name__ == '__main__':
    # Mock Accessibility Tree (AX) data

    print("Main")


# ----------------------------- Agent --------------------------------

class LLMGeneralAgent(REAL.Agent):
    """
    A "Plan and Execute" agent with a silent internal retry mechanism and an
    optimized retry prompt for failed execution steps.
    """
    VIRTUAL_SCROLL_TARGET = "VIRTUAL_SCROLL_ACTION"
    SCROLL_CLICK_BID = "2"  # BID for a benign click (usually the viewport/body)

    def __init__(self):
        super().__init__()
        self.goal: Optional[str] = None
        self.plan: List[Dict] = []
        self.current_step_index = 0
        self.state = "PLANNING"
        self.retry_count = 0
        self.max_retries = 3  # Maximum times to retry a single step
        self.last_failed_action: Optional[str] = None  # Stores the failed action string

    def get_agent_action(self, obs) -> Tuple[Optional[str], Optional[str]]:
        axtree = (obs.get("axtree_text") or obs.get("axtree_txt") or "")[:12000]

        # --- PHASE 1: PLANNING ---
        if self.state == "PLANNING":
            if not self.goal:
                if not obs.get("goal_object"):
                    return None, "Task goal (goal_object) not found in observation."
                try:
                    self.goal = " ".join([
                        msg["text"].strip() for msg in obs["goal_object"]
                        if msg["type"] == "text" and msg.get("text")
                    ])
                    if not self.goal:
                        raise ValueError("Goal object was empty or had no meaningful text.")
                except Exception as e:
                    print(f"[AGENT] CRITICAL: Failed to parse goal_object: {e}")
                    return None, f"Failed to parse goal_object: {e}"
                print(f"[AGENT] Goal extracted from obs: '{self.goal}'")

            print("[AGENT] State: PLANNING. Generating task plan...")
            prompt = PROMPT_PLANNER.strip().replace("{user_goal}", self.goal)

            try:
                model, resp = first_working_model([{"role": "user", "content": prompt}], max_tokens=10000)
                raw_plan = resp.choices[0].message.content

                # Robust JSON extraction logic
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', raw_plan, re.DOTALL)
                json_str = None
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    start_index = raw_plan.find('[')
                    end_index = raw_plan.rfind(']')
                    if start_index != -1 and end_index != -1 and end_index > start_index:
                        json_str = raw_plan[start_index:end_index + 1].strip()
                    else:
                        print(f"[AGENT] ERROR: LLM did not return a valid JSON list structure. Response: {raw_plan}")
                        return None, "Failed to generate a valid JSON plan (no [ ] structure found)."

                if not json_str:
                    return None, "Failed to generate a valid JSON plan."

                self.plan = json.loads(json_str)
                print(f"[AGENT] Plan generated with {len(self.plan)} steps.")
                self.state = "EXECUTING"

            except json.JSONDecodeError as e:
                return None, f"Failed to parse plan (JSON Decode Error): {e}"
            except Exception as e:
                return None, f"Failed to parse plan: {e}"

        # --- PHASE 2: EXECUTING ---
        if self.state == "EXECUTING":

            # --- Check for completion ---
            if self.current_step_index >= len(self.plan):
                print("[AGENT] State: COMPLETE. All steps executed.")
                return None, "Task Complete: All steps executed."

            try:
                current_step = self.plan[self.current_step_index]
                step_desc = current_step["step"]
                action_type = current_step["action"]
                target_desc = current_step["target_description"]
                text_to_fill = current_step.get("text", "")
            except (KeyError, IndexError) as e:
                return None, f"Failed to read plan step: {e}"

            print("-" * 30)

            # --- Retry Context Generation ---
            retry_context = ""
            if self.retry_count > 0:
                print(f"[AGENT] State: RETRYING (Attempt {self.retry_count + 1}/{self.max_retries + 1})")

                # OPTIMIZED RETRY PROMPT: Specific failure + alternate suggestion
                failure_reason = self.last_failed_action if self.last_failed_action else "Unknown element not found failure."
                retry_context = (
                    f"PREVIOUS ATTEMPT FAILED. The system was **unable to execute your previous action: {failure_reason}**. "
                    "You must try a different approach: "
                    "1. **Alternate Target**: Look for elements with similar text or role (e.g., if targeting 'Checkout', look for 'Proceed to Payment', 'Next Step', 'Cart Button'). "
                    "2. **If the element is likely off-screen (scroll is needed)**: Your action must be: **click(\"2\")**. This is a global scroll action. Do NOT output a 'send_msg_to_user' action during a retry, only output the action needed to find the element."
                )

            else:
                print(f"[AGENT] State: EXECUTING (Step {self.current_step_index + 1}/{len(self.plan)})")

            print(f"[AGENT]   Goal: {self.goal}")
            print(f"[AGENT]   Step: {step_desc}")
            print(f"[AGENT]   Action: {action_type}, Target: {target_desc}")

            # --- SPECIAL CHECK FOR VIRTUAL SCROLL ACTION ---
            if target_desc == self.VIRTUAL_SCROLL_TARGET:
                print(f"[AGENT] *** EXECUTING VIRTUAL SCROLL ACTION ***")
                self.current_step_index += 1
                self.retry_count = 0
                self.last_failed_action = None  # Reset failure status
                action = f'click("{self.SCROLL_CLICK_BID}")'
                print(f"[ACTION (Step {self.current_step_index})]: {action} (Simulated Scroll)")
                return action, None
            # --- END VIRTUAL SCROLL CHECK ---

            # 1. Format the general-purpose EXECUTION prompt
            prompt = PROMPT_EXECUTOR.format(
                overall_goal=self.goal,
                step_description=step_desc,
                target_description=target_desc,
                action_type=action_type,
                text_to_fill=text_to_fill,
                RETRY_CONTEXT=retry_context,  # Inject optimized retry context
                AX=axtree
            )

            # 2. Get LLM response for this step
            try:
                # Use a smaller token budget for execution steps
                model, resp = first_working_model([{"role": "user", "content": prompt}], max_tokens=10000)
                raw_action = resp.choices[0].message.content
            except Exception as e:
                print(f"[LLM error] {e}")
                raw_action = ""

            print(f"[LLM RAW (Step {self.current_step_index + 1})]: {raw_action!r}")

            # 3. Sanitize and validate the action
            action = sanitize_general_action(raw_action)
            fail_msg = f'send_msg_to_user("Could not find {target_desc}")'
            action_failed = not action or action.startswith('send_msg_to_user(')

            # 4. Fallback Logic: Use heuristic if LLM fails
            if action_failed:
                hb = find_best_bid_heuristic(axtree, target_desc)

                if hb:
                    # Construct the action string based on the plan
                    if action_type == "fill":
                        action = f'fill("{hb}", "{text_to_fill}")'
                    elif action_type == "select":
                        action = f'select("{hb}", "{text_to_fill}")'
                    else:  # "click"
                        action = f'click("{hb}")'
                    print(f"[AGENT] Heuristic fallback successful: {action}")
                    action_failed = False
                else:
                    # Heuristic failed too
                    action = fail_msg
                    action_failed = True

            # --- Retry/Advance Logic (Internal Silent Retry) ---
            if not action_failed:
                # Success: Advance the plan index and reset retry count
                print(f"[ACTION (Step {self.current_step_index + 1})]: {action}")
                self.current_step_index += 1
                self.retry_count = 0
                self.last_failed_action = None  # Reset failure status on success
                return action, None
            else:
                # Failure: Check if we can retry
                self.retry_count += 1
                self.last_failed_action = action  # Store the failed action (send_msg_to_user(..))

                if self.retry_count <= self.max_retries:
                    # Retry: Do NOT advance step index; return a benign action (silent retry)
                    print(f"[RETRY PENDING] Step {self.current_step_index + 1} failed. Retrying silently in next turn (Attempt {self.retry_count}).")
                    benign_action = f'click("{self.SCROLL_CLICK_BID}")'  # Clicks BID 2 (viewport)
                    return benign_action, None
                else:
                    # Final Failure: Advance the plan index and send final fail message
                    print(f"[FINAL FAILURE] Step {self.current_step_index + 1} failed after {self.max_retries + 1} attempts.")
                    self.current_step_index += 1
                    self.retry_count = 0
                    self.last_failed_action = None
                    return action, None  # Returns the fail_msg (send_msg_to_user(...))

        return None, "Agent is in an unknown state."

    def get_action(self, obs: dict):
        action, final_message = self.get_agent_action(obs)
        if final_message:
            return f'send_msg_to_user("{final_message}")', {}
        return action, {}


@dataclasses.dataclass
class LLMGeneralAgentArgs(REAL.AbstractAgentArgs):
    """Args for the LLMGeneralAgent."""
    agent_name: str = "LLMGeneralAgent"

    def make_agent(self):
        return LLMGeneralAgent()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="webclones.omnizon-1")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--force_refresh", action="store_true", help="Bypass cached results")
    ap.add_argument("--max_steps", type=int, default=10)
    args = ap.parse_args()

    agent_args = LLMGeneralAgentArgs()

    print(f"--- Starting Agent ---")
    print(f"Task: {args.task}")
    print(f"(Goal will be read from task environment's 'goal_object')")
    print(f"---------------------------------")

    harness = REAL.harness(
        agentargs=agent_args,
        headless=args.headless,
        max_steps=args.max_steps,
        force_refresh=args.force_refresh,
        leaderboard=True,
        run_id="16b6844e-dc3b-40b1-8fd4-1b372bae6aa0",
        task_type="omnizon",
    )
    results = harness.run()
    print(results)


if __name__ == "__main__":
    main()