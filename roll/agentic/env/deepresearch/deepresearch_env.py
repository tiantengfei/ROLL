import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Mapping

# Imports needed for the full LocalToolExecutor
from ..tools.base import BaseTool # Added
from ..tools.python_execute import PythonExecute # Added
from ..tools.terminate import Terminate # Added
from ..tools.ask_human import AskHuman # Added
from ..tools.str_replace_editor import StrReplaceEditor, ToolError as EditorToolError # Added
from ..tools.browser_use_tool_wrapper import BrowserUseToolWrapper, ToolError as BrowserToolError # Added


# Assuming BaseLanguageBasedEnv and BaseTask are structured like this:
# roll/agentic/env/base.py
# class BaseLanguageBasedEnv:
#     def __init__(self, *args, **kwargs): pass
#     def reset(self, *args, **kwargs): raise NotImplementedError
#     def step(self, action: str): raise NotImplementedError
#     async def close(self): pass

# roll/agentic/controller.py
# class BaseTask:
#     def __init__(self, *args, **kwargs): pass
#     async def close(self): pass

# Corrected imports based on the problem description
from ..base import BaseLanguageBasedEnv
try:
    from ...controller import BaseTask # Try to import from the expected location
except ImportError:
    # Fallback if the controller is in a different location or for placeholder
    # This is a common pattern if the exact structure isn't known.
    # For this task, we'll assume a placeholder if direct import fails.
    class BaseTask: # Placeholder
        def __init__(self, client_args: Mapping[str, Any], n_clients: int, env_client_cls: Any, env_name: str, *args, **kwargs):
            self.clients = [] # Simplified
            pass
        async def close(self):
            for client in self.clients:
                if hasattr(client, 'close') and asyncio.iscoroutinefunction(client.close):
                    await client.close()
            self.clients.clear()


from .config import DeepResearchEnvConfig

# Global constants (to be preserved)
SYSTEM_PROMPT = (
    "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, web browsing, or human interaction (only for extreme cases), you can handle it all."
    "The initial directory is: {directory}"
)

NEXT_STEP_PROMPT = """
Based on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.

If you want to stop the interaction at any point, use the `terminate` tool/function call.
"""

SUMMARY_HISTORY_PROMPT = """用户请求了解的信息如下：
{0}。

当前采用分步方式完成任务。在此步骤之前，历史已提炼的内容包括：
{1}。

当前步骤获取到的内容如下：
{2}。

现在你是一位内容理解专家，负责信息内容的理解，内容筛选，内容总结。
现在请你
1、理解用户的请求，并对获取的所有信息（历史内容 + 当前步骤内容）进行判断和筛选，保留与用户原始请求最相关、最核心的内容。例如，对于“山西13天古建游”，整合所有已获取的景点推荐、路线建议、交通信息估算等。
2. 对所有操作步骤进行总结，如果某个核心查询（如“山西古建景点推荐”、“太原到大同交通方式”）在历史步骤中已执行过，明确标注。
3. **关键判断:** 基于整合后的信息，**明确判断**当前用户请求 '{0}' 所需的核心信息（例如，生成建议性Timeline所需的景点列表、路线逻辑、交通估算等）是否**已经基本足够**在“已获取的信息”中找到。请在总结末尾使用以下标记之一：
    *   `[Data Sufficient for Planning]`：如果信息已足够支撑下一步的Timeline生成。
    *   `[Data Insufficient for Planning]`：如果仍需通过工具（主要是WebSearch）获取关键信息才能生成Timeline。

要求：
1、只根据当前步骤的内容和历史已提炼的内容来得到最终结果，不要新增其他东西。
2、过滤掉明显不合理、重复冗余的查询过程描述，专注于已获取的事实信息和规划所需要素。
3、返回形式为：
已获取的信息：[整合后的核心信息列表，用于规划Timeline]
已进行的操作步骤：[步骤总结，标注重复查询]
信息判断：[Data Sufficient for Planning] 或 [Data Insufficient for Planning]

注意：只返回总结后的内容即可，不用返回你的思考过程。
"""

# ToolError class (to be preserved)
class ToolError(Exception): # General error for the executor
    pass

# LocalToolExecutor class (to be preserved - basic structure)
class LocalToolExecutor:
    def __init__(self, task_description: str, config: Optional[Dict[str, Any]] = None): # Modified config param name
        self.task_description = task_description
        self.config = config if config else {} # Modified to use self.config
        self.max_steps = self.config.get("max_steps", 20)

        browser_tool_config = self.config.get("browser_tool_config", {})
        str_editor_workspace_root = self.config.get("str_editor_workspace_root", None)

        self.python_tool = PythonExecute()
        self.terminate_tool = Terminate()
        self.ask_human_tool = AskHuman()
        self.str_replace_editor_tool = StrReplaceEditor(workspace_root=str_editor_workspace_root)
        self.browser_tool = BrowserUseToolWrapper(browser_config_args=browser_tool_config)

        self.tools: Dict[str, BaseTool] = {
            self.python_tool.name: self.python_tool,
            self.terminate_tool.name: self.terminate_tool,
            self.ask_human_tool.name: self.ask_human_tool,
            self.str_replace_editor_tool.name: self.str_replace_editor_tool,
            self.browser_tool.name: self.browser_tool,
        }

        base_system_message = "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, web browsing, or human interaction (only for extreme cases), you can handle it all."
        core_message = base_system_message

        tools_section_str = ""
        if self.tools:
            tools_list_parts = []
            for tool in self.tools.values():
                schema_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                tools_list_parts.append(json.dumps(schema_dict))

            tools_details = "\n".join(tools_list_parts)

            tools_section_str = (
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                f"{tools_details}\n"
                "</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            )

        self.system_prompt_value = (
            f"<|im_start|>system\n{core_message}{tools_section_str}<|im_end|>\n"
        )

        self.current_observation_text = self.system_prompt_value

        self.current_step: int = 0
        self.task_completed: bool = False
        self.latest_status_info: Dict[str, Any] = {"message": "Session initialized."}
        self.next_prompt_value = self.config.get("next_prompt_value", NEXT_STEP_PROMPT)

        # logging.info is used in the new DeepResearchEnv, print is from old file.
        # Using logging.info for consistency with the rest of the new file.
        logging.info(f"[LocalToolExecutor] Initialized for task: '{self.task_description}'. Max steps: {self.max_steps}.")
        logging.info(f"[LocalToolExecutor] Available tools: {list(self.tools.keys())}")
        logging.info(f"[LocalToolExecutor] System Prompt: {self.system_prompt_value[:500]}...")

    def get_initial_observation(self) -> str:
        return self.current_observation_text

    async def process_action(self, actions: List[str]) -> None:
        self.current_step += 1
        tool_executed_successfully_overall = False
        observation_parts = []

        if self.task_completed:
            self.current_observation_text = "Tried to act on a completed task. No state change."
            logging.warning("[LocalToolExecutor] Warning: Action processed on already completed task.")
            return

        if not actions:
            self.current_observation_text = "No actions were performed."
            self.latest_status_info = {"success": True, "message": "Empty action list processed."}
            if not self.task_completed and self.current_step >= self.max_steps:
                self.task_completed = True
                self.current_observation_text += "\nMax steps reached. Episode terminated."
                self.latest_status_info = {"success": False, "message": "Terminated due to max steps with empty action."}
            logging.info(f"[LocalToolExecutor] Step {self.current_step}: Action(s)='{actions}', Done={self.task_completed}")
            logging.info(f"[LocalToolExecutor] Observation: {self.current_observation_text[:200]}...")
            return

        self.latest_status_info = {"success": False, "message": "Processing actions."}

        for i, action_json_string in enumerate(actions):
            if self.task_completed:
                break

            action_item = None
            tool_name = "UnknownTool"
            args_for_log = {}
            tool_executed_successfully_item = False

            processed_action_json_string = action_json_string.strip()
            original_action_for_error_msg = processed_action_json_string # Save for error messages

            if processed_action_json_string.startswith("<tool_call>") and processed_action_json_string.endswith("</tool_call>"):
                content_start_index = processed_action_json_string.find(">") + 1
                content_end_index = processed_action_json_string.rfind("<")
                if content_start_index < content_end_index: # Basic check
                    processed_action_json_string = processed_action_json_string[content_start_index:content_end_index].strip()

            try:
                action_item = json.loads(processed_action_json_string)
                if not isinstance(action_item, dict):
                    # Use original_action_for_error_msg here if processed_action_json_string is the result of extraction
                    raise ToolError(f"Parsed action content is not a dictionary. Original action string part: '{processed_action_json_string}'")

                tool_name = action_item.get("name")
                if not tool_name:
                    error_message = f"Error: Missing 'name' in action content: {processed_action_json_string}"
                    observation_parts.append(f"{i+1}. {error_message}") # This error is about malformed action, not a tool response yet
                    self.latest_status_info = {"success": False, "message": "Missing 'name' in parsed action."}
                    continue

                args = action_item.get("arguments")
                if args is None: args = {}
                elif not isinstance(args, dict):
                    logging.warning(f"[LocalToolExecutor] Warning: 'arguments' in action is not a dict. Action: {action_json_string}")
                args_for_log = args

                calling_tool_line = f"Calling tool {tool_name}: {json.dumps(args_for_log)}"

                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = await tool.execute(**args)
                    tool_executed_successfully_item = True
                    self.latest_status_info = {"success": True, "message": f"Tool {tool_name} executed."}
                    # Wrap successful tool execution in <tool_response>
                    tool_output_narrative = f"Successfully executed tool {tool_name}.\nResult: {tool_result}"
                    formatted_action_observation = f"<tool_response>\n{tool_output_narrative}\n</tool_response>"
                    observation_parts.append(formatted_action_observation)

                    if tool_name == self.terminate_tool.name:
                        self.task_completed = True
                        status_arg = args.get("status", "success")
                        self.latest_status_info = {"success": status_arg == "success", "message": f"Terminated by agent with status: {status_arg}"}
                        break
                else:
                    error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"
                    # This error is about an attempt to call a non-existent tool, so it's a kind of "tool response" to that attempt.
                    error_message_details = f"Error: Attempted to call unknown tool '{tool_name}'."
                    formatted_action_observation = f"<tool_response>\n{error_message_details}\n</tool_response>"
                    observation_parts.append(formatted_action_observation)
                    self.latest_status_info = {"success": False, "message": f"Unknown tool: {tool_name}."}

            except json.JSONDecodeError:
                error_message = f"Could not parse action JSON content: {processed_action_json_string}. Original string: {original_action_for_error_msg}"
                # This error is about malformed action, not a direct tool response to a valid call attempt. Keep plain for now.
                observation_parts.append(f"{i+1}. Error: {error_message}")
                self.latest_status_info = {"success": False, "message": "Action JSON parsing error."}
                continue
            except ToolError as e: # Covers missing function_name if not caught earlier, or other validation issues
                error_message = str(e)
                # This is a ToolError raised by our own validation, like non-dict parsed action. Keep plain.
                if 'calling_tool_line' not in locals():
                    calling_tool_line = f"Attempted call with malformed action: {original_action_for_error_msg}"
                formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: {error_message}" # Keep plain
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Invalid action item: {error_message}"}
            except (BrowserToolError, EditorToolError) as e:
                error_message = str(e)
                if 'calling_tool_line' not in locals(): calling_tool_line = f"Calling tool {tool_name} (args: {args_for_log})"
                error_message_details = f"Error executing tool {tool_name}: {error_message}"
                formatted_action_observation = f"<tool_response>\n{error_message_details}\n</tool_response>"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Tool error with {tool_name}: {error_message}"}
            except TypeError as e: # Typically an issue with tool arguments
                error_message = str(e)
                if 'calling_tool_line' not in locals(): calling_tool_line = f"Calling tool {tool_name} (args: {args_for_log})"
                error_message_details = f"Error: Argument mismatch for tool {tool_name}. Details: {error_message}"
                formatted_action_observation = f"<tool_response>\n{error_message_details}\n</tool_response>"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Tool argument mismatch for {tool_name}: {error_message}"}
            except Exception as e: # Other unexpected errors during tool execution
                error_message = str(e)
                if 'calling_tool_line' not in locals(): calling_tool_line = f"Calling tool {tool_name} (args: {args_for_log})"
                error_message_details = f"Error: Unexpected issue executing tool {tool_name}. Details: {error_message}"
                formatted_action_observation = f"<tool_response>\n{error_message_details}\n</tool_response>"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Unexpected error with tool {tool_name}: {error_message}"}

            if tool_executed_successfully_item:
                tool_executed_successfully_overall = True

        if observation_parts:
            self.current_observation_text = "\n\n".join(observation_parts)
        elif not actions:
             self.current_observation_text = "No actions were performed."
        else:
              if self.latest_status_info.get("message") == "Processing actions.":
                   self.current_observation_text = "Actions processed, but no specific observations were generated."

        if not self.task_completed and self.current_step >= self.max_steps:
            self.task_completed = True
            self.current_observation_text += "\n\nMax steps reached. Episode terminated."
            self.latest_status_info = {"success": False, "message": "Terminated due to max steps."}

        if not self.task_completed and self.latest_status_info.get("message") == "Processing actions.":
            if tool_executed_successfully_overall:
                self.latest_status_info = {"success": True, "message": "Tool(s) executed."}
            elif not tool_executed_successfully_overall : # Corrected condition
                 self.latest_status_info = {"success": False, "message": "No tool executed successfully or actions resulted in no specific outcome."}

        logging.info(f"[LocalToolExecutor] Step {self.current_step}: Action(s)='{actions}', Done={self.task_completed}")
        logging.info(f"[LocalToolExecutor] Observation: {self.current_observation_text[:200]}...")

    def get_current_observation(self) -> str: # Added from original to complete the class
        return self.current_observation_text

    def is_done(self) -> bool:
        return self.task_completed

    def get_reward(self) -> float:
        if not self.task_completed:
            return self.config.get("reward_step", 0.0)

        if self.latest_status_info.get("success", False):
            return self.config.get("reward_success", 1.0)

        message = self.latest_status_info.get("message", "").lower()
        if "max steps" in message:
             return self.config.get("reward_timeout", -1.0)
        return self.config.get("reward_failure", -0.5)

    # Removed @property for latest_status_info to match provided original LocalToolExecutor structure
    # if it's not a property in the original. Assuming it's a direct attribute access if needed.
    # However, the placeholder had it as a property. Sticking to original file's direct access.
    # latest_status_info is accessed as self.latest_status_info directly in DeepResearchEnv
    # Re-checking original: it's a direct attribute, not a property.

    async def cleanup(self) -> None:
        logging.info("[LocalToolExecutor] Cleaning up resources...") # Changed print to logging.info
        if hasattr(self.browser_tool, 'cleanup') and callable(self.browser_tool.cleanup):
            await self.browser_tool.cleanup()
        logging.info("[LocalToolExecutor] Cleanup finished.") # Changed print to logging.info


# Refactored class: OpenManusLocalEnvClient -> DeepResearchEnv
class DeepResearchEnv(BaseLanguageBasedEnv):
    def __init__(self, config: DeepResearchEnvConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.tool_executor: Optional[LocalToolExecutor] = None

        # executor_config_dict is now effectively self.config itself,
        # as LocalToolExecutor now expects the full config object (or a dict derived from it)
        # and accesses fields like browser_tool_config, str_editor_workspace_root directly.
        # The original LocalToolExecutor took a dict named 'config'.
        # The new DeepResearchEnvConfig is a class.
        # We need to pass the necessary parts of DeepResearchEnvConfig to LocalToolExecutor's 'config' param.
        # Let's adapt to pass a dictionary derived from DeepResearchEnvConfig to LocalToolExecutor.
        self.executor_config_dict: Dict[str, Any] = {
            "max_steps": self.config.max_steps,
            "reward_step": self.config.reward_step,
            "reward_success": self.config.reward_success,
            "reward_failure": self.config.reward_failure,
            "reward_timeout": self.config.reward_timeout,
            "invalid_act_score": self.config.invalid_act_score,
            # Pass other relevant attributes from self.config that LocalToolExecutor might expect
            # For example, if LocalToolExecutor uses browser_tool_config:
            # "browser_tool_config": getattr(self.config, 'browser_tool_config', {}),
            # "str_editor_workspace_root": getattr(self.config, 'str_editor_workspace_root', None),
        }
        # Add fields from additional_config if they exist
        if hasattr(self.config, 'additional_config') and self.config.additional_config:
             self.executor_config_dict.update(self.config.additional_config)
        # LocalToolExecutor expects 'browser_tool_config' and 'str_editor_workspace_root'
        # to be top-level keys in its 'config' dictionary.
        # We need to ensure these are correctly populated from DeepResearchEnvConfig.
        # The DeepResearchEnvConfig has these as commented-out direct attributes.
        # If they are defined on DeepResearchEnvConfig, they should be passed.
        # Example:
        if hasattr(self.config, 'browser_tool_config'): # Check if attr exists
            self.executor_config_dict["browser_tool_config"] = self.config.browser_tool_config
        if hasattr(self.config, 'str_editor_workspace_root'): # Check if attr exists
            self.executor_config_dict["str_editor_workspace_root"] = self.config.str_editor_workspace_root


    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, str]: # Return type changed

        actual_task_desc = kwargs.get("task_description", "Default research task.") # Renamed for clarity

        if self.tool_executor:
            logging.info("Cleaning up previous LocalToolExecutor instance.")
            try:
                # Ensure cleanup is awaited if in an event loop, or run if not.
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(self.tool_executor.cleanup()) # Fire and forget if not awaiting reset
            except RuntimeError:
                asyncio.run(self.tool_executor.cleanup())
            self.tool_executor = None

        self.tool_executor = LocalToolExecutor(
            task_description=actual_task_desc, # Use actual_task_desc
            config=self.executor_config_dict
        )
        initial_obs_string = self.tool_executor.get_initial_observation()
        return {"initial_observation": initial_obs_string, "task_description": actual_task_desc}

    def get_system_prompt(self) -> Optional[str]:
        """Returns the system prompt from the active tool executor, if available."""
        if self.tool_executor and hasattr(self.tool_executor, 'system_prompt_value'):
            return self.tool_executor.system_prompt_value
        logging.warning("[DeepResearchEnv.get_system_prompt] Tool executor not initialized or does not have system_prompt_value.")
        return None

    def render(self, mode: str = "text") -> str:
        if self.tool_executor:
            return self.tool_executor.get_current_observation()
        return "Environment not yet reset or tool executor not initialized."

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        if not self.tool_executor:
            error_msg = "Tool executor not initialized. Call reset() first."
            logging.error(error_msg)
            return (
                {"error": error_msg},
                self.config.reward_failure,  # Default failure reward
                True,  # Done due to error
                {"error": error_msg, "status": "failure"}
            )

        actions_to_execute: List[str] = []
        if action and action.strip(): # Check if action string is not None or empty/whitespace
            try:
                parsed_json = json.loads(action)
                if isinstance(parsed_json, list):
                    # Further validation: ensure all elements in the list are strings
                    if all(isinstance(item, str) for item in parsed_json):
                        actions_to_execute = parsed_json
                    else:
                        logging.warning(f"[DeepResearchEnv.step] Parsed action JSON is a list, but not all items are strings: {action[:200]}")
                        # actions_to_execute remains []
                else:
                    logging.warning(f"[DeepResearchEnv.step] Parsed action JSON is not a list: {action[:200]}")
                    # actions_to_execute remains []
            except json.JSONDecodeError:
                logging.warning(f"[DeepResearchEnv.step] Failed to decode action string as JSON. Treating as no-op/narrative: {action[:200]}")
                # actions_to_execute remains []
        else:
            logging.info("[DeepResearchEnv.step] Received empty or null action string. No actions to execute.")
            # actions_to_execute remains []

        try:
            # Attempt to run process_action.
            try:
                loop = asyncio.get_running_loop()
                # If in an event loop, we can't just call asyncio.run().
                # This part is tricky. For now, we assume ensure_future is part of the solution
                # if the environment's step method is itself called from an async context that manages the loop.
                # If step is called synchronously, asyncio.run() is the way to run an async method.
                future = asyncio.ensure_future(self.tool_executor.process_action(actions_to_execute))
                asyncio.run(future)
            except RuntimeError as e:
                if "cannot call run() while another loop is running" in str(e):
                    # For now, we'll log and re-raise or handle gracefully.
                    logging.warning(f"Asyncio RuntimeError in step: {e}. This needs careful handling.")
                    # If we need to proceed, we'd need a way to run the async code.
                    # For now, let's assume it implies an issue that stops the step.
                    # This part is highly dependent on the execution context of the `step` method.
                    # A placeholder for what might happen if an action can't be processed due to async issues:
                    return (
                        self.tool_executor.get_current_observation(), # Previous observation
                        self.config.reward_failure, # Penalize
                        True, # End episode
                        {"error": "Async execution error in step", "status": "failure"}
                    )
                else:
                    raise # Re-raise other RuntimeErrors
            except ToolError as e:
                logging.warning(f"ToolError during step: {e}")
                # Observation, reward, done, info
                observation = self.tool_executor.get_current_observation()
                reward = self.tool_executor.get_reward() # Should reflect failure
                done = self.tool_executor.is_done()
                info = self.tool_executor.latest_status_info
                info["error"] = str(e)
                return observation, reward, done, info


            observation = self.tool_executor.get_current_observation()
            reward = self.tool_executor.get_reward()
            done = self.tool_executor.is_done()
            info = self.tool_executor.latest_status_info.copy() # Get a copy

        except Exception as e:
            logging.exception(f"Unexpected error during step: {e}")
            return (
                {"error": str(e)},
                self.config.reward_failure,
                True,
                {"error": str(e), "status": "failure"}
            )

        return observation, reward, done, info

    async def close(self) -> None:
        if self.tool_executor:
            await self.tool_executor.cleanup()
            self.tool_executor = None
        # Call super().close() if BaseLanguageBasedEnv has its own async close logic
        # await super().close()

# Refactored class: OpenManusLocalTask -> DeepResearchTask
class DeepResearchTask(BaseTask):
    env_client_cls: Type[DeepResearchEnv] = DeepResearchEnv
    env_name: str = "deepresearch"

    def __init__(self, client_args: Mapping[str, Any], n_clients: int, *args, **kwargs):
        # Ensure client_args contains a config of the correct type
        if "config" not in client_args or not isinstance(client_args["config"], DeepResearchEnvConfig):
            raise ValueError(
                "client_args must contain a 'config' key with a DeepResearchEnvConfig instance."
            )

        # Pass env_client_cls and env_name to BaseTask constructor
        super().__init__(
            client_args=client_args,
            n_clients=n_clients,
            env_client_cls=self.env_client_cls, # Pass the class itself
            env_name=self.env_name,
            *args, **kwargs
        )
        # BaseTask.__init__ should handle client creation.
        # If BaseTask does not create clients, they would be created here:
        # self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]


    async def close(self) -> None:
        # Use the close mechanism from the placeholder BaseTask or a real one
        await super().close()
        # If super().close() doesn't handle individual client closing, do it here:
        # for client in self.clients:
        #     if hasattr(client, 'close') and asyncio.iscoroutinefunction(client.close):
        #         await client.close()
        # self.clients.clear()
        logging.info("DeepResearchTask and its clients closed.")

# Example of how to use (optional, for testing or demonstration)
# async def main():
#     config = DeepResearchEnvConfig(max_steps=10)
#     env = DeepResearchEnv(config=config)
#     obs = env.reset(task_description="Find information about async Python.")
#     print(f"Initial observation: {obs}")

#     action1 = "Search for 'asyncio best practices'"
#     obs, reward, done, info = env.step(action1)
#     print(f"Action: {action1}\nObservation: {obs}\nReward: {reward}\nDone: {done}\nInfo: {info}\n")

#     action2 = "Simulate success by including 'success' in the action"
#     obs, reward, done, info = env.step(action2)
#     print(f"Action: {action2}\nObservation: {obs}\nReward: {reward}\nDone: {done}\nInfo: {info}\n")

#     await env.close()

#     # Example for DeepResearchTask
#     task_config = DeepResearchEnvConfig(max_steps=5)
#     task = DeepResearchTask(client_args={"config": task_config}, n_clients=1)
#     # To interact with the task's envs, you'd typically access them via task.clients
#     if task.clients:
#         client_env = task.clients[0]
#         obs = client_env.reset(task_description="Test task in DeepResearchTask")
#         print(f"Task Env Initial observation: {obs}")
#         # ... interact further ...
#     await task.close()

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # asyncio.run(main())
