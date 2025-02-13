import time

MAX_TOKENS = 1200
# GPT_MODEL = "gpt-4o"


def encap_msg(msg, role='user', **kwargs):
    dialog = {'role': role, 'content': msg}
    dialog.update(kwargs)
    return dialog

def get_gpt_completion(model_name,dialogs, client, tools=None, tool_choice=None, temperature=0.6, max_tokens=MAX_TOKENS):
    max_retries = 10 #5 #1
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name, #"gpt-3.5-turbo-1106",
                messages=dialogs,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            #this_cost = prompt_tokens / 1e6 * PROMPT_COST_1M + completion_tokens / 1e6 * COMPLETION_COST_1M
            msg = response.choices[0].message.content
            if(msg==''):
                print('Fail! Please retry')
                return 1, '', 0, 0
            else:
                return 0, msg, prompt_tokens, completion_tokens
        except Exception as e:
            # 检查是否是因为消息长度超过限制
            if "context length" in str(e) or "too long" or "max_total_tokens" in str(e):  # 假设API会抛出包含"context length"的错误信息
                print('Input too long! Please retry')
                return 1, '', 0, 0
            if i < max_retries - 1:
                time.sleep(8)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"