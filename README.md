# QA对构建

## ASR
FUNASR.py

## 生成QA对
prompt：data_optimize_qa.txt

## 生成prompt
prompt：data_final_qa.txt

QA的构建调用data_get_qa.py，修改输入和输出，更换对应prompt的即可。
INPUT_DIR = ""  # 输入的 txt 或 json 文件目录
OUTPUT_DIR = ""  # 结果输出目录
PROMPT_TEMPLATE_PATH = ""  # Prompt 模板路径

没有设置args，有需要可以自己添加下

采用并发执行
MAX_LINES_PER_FILE = 500  # 拆分文本的行数限制
MAX_CONCURRENT_TASKS = 10  # 最大并发任务数

MAX_LINES_PER_FILE主要用来控制文本长度，避免大模型生成结果过长，导致未完整生成结果，json格式不准确，无法保存下生成结果。
