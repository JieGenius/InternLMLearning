### xtuner 入门：
xtuner安装
```
git clone -b v0.1.9  https://github.com/InternLM/xtuner
cd xtuner
pip install -e '.[all]'
```
列出配置
```
xtuner list-cfg
```
选择配置并复制到指定目录
```bash
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```
修改配置：
data_path，epoch，batch_size，数据导入的map函数等等，模型路径
```
vim ***
pretrained_model_name_or_path = xxx
data_path = xxx
```
训练
```bash
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2
```
模型转换
```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
模型merge
```bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
```
模型对话
```bash
xtuner chat ./merged --prompt-template internlm_chat --bits 4
```
### jsonL
JSONL 文件（JSON Lines），则是一种每行包含一个独立的 JSON 对象的文本文件格式。每行都是一个有效的 JSON 对象，使用换行符分隔。相比于 JSON 文件，JSONL 文件更加轻量，每行为独立的 JSON 对象，没有逗号或其他分隔符。
example:
```json lines
{"name": "John", "age": 30}
{"name": "Jane", "age": 25}
{"name": "Bob", "age": 40}
```
### 大佬chatGPT prompt 例子
有很强的参考和学习意义，特此记录
```
Write a python file for me. using openpyxl. input file name is MedQA2019.xlsx
Step1: The input file is .xlsx. Exact the column A and column D in the sheet named "DrugQA" .
Step2: Put each value in column A into each "input" of each "conversation". Put each value in column D into each "output" of each "conversation".
Step3: The output file is .jsonL. It looks like:
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
Step4: All "system" value changes to "You are a professional, highly experienced doctor professor. You always provide accurate, comprehensive, and detailed answers based on the patients' questions."
```
随机划分训练验证集
```
my .jsonL file looks like:
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
Step1, read the .jsonL file.
Step2, count the amount of the "conversation" elements.
Step3, randomly split all "conversation" elements by 7:3. Targeted structure is same as the input.
Step4, save the 7/10 part as train.jsonl. save the 3/10 part as test.jsonl
```