import json
import json5
import os
import re
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

# ========== 配置 ========== #
INPUT_DIR = ""  # 输入的 txt 或 json 文件目录
OUTPUT_DIR = ""  # 结果输出目录
PROMPT_TEMPLATE_PATH = ""  # Prompt 模板路径
MODEL_NAME = "deepseek-v3"  # 使用的 LLM 模型
MAX_LINES_PER_FILE = 500  # 拆分文本的行数限制
MAX_CONCURRENT_TASKS = 10  # 最大并发任务数

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 初始化 OpenAI 客户端 ========== #
model_client_map = {
    MODEL_NAME: OpenAI(
        api_key="",
        base_url="https://api.zhizengzeng.com/v1",
    )
}

# ========== 读取并解析 JSON 文件 ========== #
def read_json_file(file_path):
    """读取 JSON 文件并返回内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"❌ 无法读取 JSON 文件 {file_path}，错误: {e}")
        return None

# ========== 按行数拆分文本文件函数（并带监控进度） ========== #
def split_txt_file(file_path, max_lines=MAX_LINES_PER_FILE):
    base_name, ext = os.path.splitext(os.path.basename(file_path))

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}，错误: {e}")
        return []

    total_lines = len(lines)

    if total_lines <= max_lines:
        print(f"📄 文件 {file_path} 内容小于 {max_lines} 行，无需拆分")
        return [''.join(lines)]

    num_parts = (total_lines // max_lines) + 1

    split_contents = []
    for i in range(num_parts):
        start_index = i * max_lines
        end_index = min((i + 1) * max_lines, total_lines)
        part_content = ''.join(lines[start_index:end_index])
        split_contents.append(part_content)

    return split_contents

# ========== 读取 Prompt 模板 ========== #
def read_prompt_template():
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ========== 读取并拆分输入目录中的 txt 和 json 文件（并带进度监控） ========== #
def read_and_split_files():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt") or f.endswith(".json")]
    file_contents = {}

    # 使用并发方式处理文件拆分
    with ThreadPoolExecutor() as executor:
        futures = {
            file: executor.submit(process_file, os.path.join(INPUT_DIR, file))
            for file in files
        }

        # 使用 tqdm 进度条监控拆分进度
        for file, future in tqdm(futures.items(), desc="处理文件中", total=len(futures)):
            split_contents = future.result()
            base_name = os.path.splitext(file)[0]  # 去掉文件扩展名
            for idx, content in enumerate(split_contents):
                new_filename = f"{base_name}_{idx + 1}.txt"  # 新文件名：原文档名_序号.txt
                file_contents[new_filename] = content

    return file_contents

def process_file(file_path):
    """处理文件（txt 或 json）"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return split_txt_file(file_path, MAX_LINES_PER_FILE)
    elif ext == ".json":
        content = read_json_file(file_path)
        if content:
            return [json.dumps(content, ensure_ascii=False)]  # 将 JSON 内容转为字符串
        else:
            return []
    return []

# ========== 检查输出目录是否已有结果（如果已存在且非空，则跳过） ========== #
def check_if_result_exists(filename):
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                result = json.load(f)
                if result == {}:  # 如果结果是空字典，继续处理
                    print(f"📁 `{filename}` 结果为空，继续处理。")
                    return False
                elif result:  # 如果结果非空，则跳过
                    print(f"📁 `{filename}` 已有有效结果，跳过处理。")
                    return True
            except json.JSONDecodeError:
                pass  # 如果无法解析 JSON，跳过

    return False

# ========== 生成 LLM 消息格式（并发处理） ========== #
async def generate_message_for_file(filename, content, prompt_template, progress_bar, semaphore):
    async with semaphore:
        # 更新为新的 JSON 文件路径
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)

        # 替换 <txt> 部分并生成完整的 prompt
        prompt = prompt_template.replace("<txt>", content)

        progress_bar.update(1)  # 更新进度条
        return filename, [{"role": "user", "content": prompt}]


async def generate_messages_concurrently(prompt_template, txt_contents, semaphore):
    tasks = []
    with tqdm(total=len(txt_contents), desc="生成消息中") as progress_bar:
        for filename, content in txt_contents.items():
            tasks.append(generate_message_for_file(filename, content, prompt_template, progress_bar, semaphore))

        # 并发执行所有任务
        messages = await asyncio.gather(*tasks)

    # 将结果转换成字典形式
    messages_dict = {filename: message for filename, message in messages}

    return messages_dict

# ========== 清理并解析 LLM 返回结果（支持并发） ========== #
async def clean_and_parse_llm_result(raw_result: str):
    """ 清理 LLM 生成的 JSON 结果，并解析为 Python 字典 """

    if not raw_result.strip():
        logging.warning("⚠️ LLM 结果为空！")
        return {}

    # 移除 Markdown 代码块 ```json ... ```
    raw_result = re.sub(r"^```json\s*|\s*```$", "", raw_result.strip())

    try:
        # 优先使用标准 JSON 解析
        return json.loads(raw_result)
    except json.JSONDecodeError as e:
        logging.warning(f"⚠️ 标准 JSON 解析失败: {e}")

    try:
        # 尝试使用 json5 解析（支持更宽松的 JSON 语法）
        return json5.loads(raw_result)
    except Exception as e:
        logging.error(f"❌ JSON5 解析仍然失败: {e}")
        return {}

# ========== 异步调用单个 LLM 并实时保存结果 ========== #
async def call_single_llm_json_async(task_id, filename, messages, model, progress_bar, semaphore):
    async with semaphore:
        max_retries = 3
        retry_count = 0

        # 检查文件是否已有结果，如果已有结果则跳过
        if check_if_result_exists(filename):
            progress_bar.update(1)  # 跳过的文件更新进度条
            return filename, ""  # 跳过当前文件

        print(f"🟡 任务 {task_id}: 开始调用 {model} 处理 `{filename}`...")

        while retry_count < max_retries:
            try:
                response = await asyncio.to_thread(
                    model_client_map[model].chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.9
                )

                if response.choices and len(response.choices) > 0:
                    llm_output = response.choices[0].message.content.strip()
                    print(f"✅ 任务 {task_id}: `{filename}` 处理完成")
                    progress_bar.update(1)

                    # 清理并保存当前子任务的结果
                    cleaned_result = await clean_and_parse_llm_result(llm_output)
                    base_filename = os.path.splitext(filename)[0]  # 去掉文件扩展名
                    output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")  # 保存为 .json 文件
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_result, f, ensure_ascii=False, indent=4)
                    print(f"📁 子任务结果已保存: {output_path}")
                    return filename, llm_output

                else:
                    print(f"⚠️ 任务 {task_id}: `{filename}` 响应为空")
                    progress_bar.update(1)
                    return filename, ""

            except json.JSONDecodeError as e:
                retry_count += 1
                print(f'{model} 返回错误 JSON 格式数据，第 {retry_count} 次重试...')
                time.sleep(1)  # 延迟 1 秒后重试
                break

            except Exception as e:
                retry_count += 1
                print(f"🔴 任务 {task_id}: `{filename}` 调用失败，重试 {retry_count}/{max_retries} 次: {e}")
                await asyncio.sleep(2)  # 等待一段时间后再重试
                break

        print(f"⚠️ 任务 {task_id}: `{filename}` 处理失败")
        progress_bar.update(1)
        return filename, ""

# ========== 并发调用 LLM API（并监控进度） ========== #
async def call_llm_json_parallel_async(messages_dict, model, semaphore):
    tasks = []
    with tqdm(total=len(messages_dict), desc="调用 LLM 中...") as progress_bar:
        for task_id, (filename, messages) in enumerate(messages_dict.items()):
            task = asyncio.create_task(call_single_llm_json_async(task_id + 1, filename, messages, model, progress_bar, semaphore))
            tasks.append(task)

        # 并发执行所有任务
        results = await asyncio.gather(*tasks)

    return results

# ========== 主程序 ========== #
async def main():
    prompt_template = read_prompt_template()
    txt_contents = read_and_split_files()

    if not txt_contents:
        print("❌ 未找到任何 txt 或 json 文件！")
        return

    # 创建并发信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # 并发生成 messages，并加入进度监控
    messages_dict = await generate_messages_concurrently(prompt_template, txt_contents, semaphore)

    # 并发调用 LLM（监控进度）
    await call_llm_json_parallel_async(messages_dict, MODEL_NAME, semaphore)

if __name__ == "__main__":
    asyncio.run(main())
