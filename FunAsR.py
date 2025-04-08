import os
from funasr import AutoModel
from tqdm import tqdm  # 导入 tqdm 库

# 加载 FunASR 语音识别模型
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")

# 指定音频文件夹
input_folder = "assets/t"
output_folder = "output"  # 指定输出文件夹

# 确保 output 文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 获取所有 WAV 文件
wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]

# 使用 tqdm 显示进度条
for file_name in tqdm(wav_files, desc="处理文件", unit="文件"):
    input_file = os.path.join(input_folder, file_name)  # 完整路径

    # 进行语音识别
    res = model.generate(input=input_file, batch_size_s=300, hotword='魔搭')

    # 获取文件名（无扩展名）
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}.txt")  # 输出路径改为 output 文件夹

    # 保存识别结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(res[0]["text"] if isinstance(res, list) and "text" in res[0] else str(res))

    # 更新进度条
    tqdm.write(f"识别结果已保存到 {output_path}")

print("所有文件处理完成！")
