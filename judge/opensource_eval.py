import requests
import json
import traceback
import json
import traceback
import re
import sys
from time import sleep
from tqdm.asyncio import tqdm
from multiprocessing import Pool
import numpy as np  
from openai import OpenAI
import pandas as pd
import os
import base64
import random
os.environ["OPENAI_API_KEY"] = ""
base_url=""
judge_prompt = """
您是评判图文问答对正确性的专家，请根据给定问题、图片，标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。
首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：

```
问题：图中这位总统的孩子叫什么名字？
图片：https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：图中这位总统（巴拉克·奥巴马）有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：图中这位总统的孩子叫什么名字？
图片：https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：图中这位总统没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出图中这位总统有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：图中这位总统的孩子叫什么名字？
图片：https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道图中这位总统有两个孩子。
模型预测4：图中这位总统（巴拉克·奥巴马）有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
模型预测5：抱歉，我无法直接查看图片内容。请提供更多关于图片或总统的详细信息，我将尽力帮助你得到孩子的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“图中的大桥的全长是多少米？”和图片"https://upload.wikimedia.org/wikipedia/commons/8/83/Songpu_Bridge_2.JPG"，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。 
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“图中的矿物的主要化学成分是什么？和图片"https://upload.wikimedia.org/wikipedia/commons/c/cf/Magnesite-t06-203a.jpg"， ”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
图片：{url}
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()

def call_model(messages, modelname):
    k = 3
    ouput = ""
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=base_url,
            )
            completion = client.chat.completions.create(
                model=modelname,
                messages=messages
            )
            ouput = completion.choices[0].message.content
            if ouput != None and ouput != "":
                break
        except Exception as e:
            print(e)
            continue
    return ouput, None

def write_to_file(info):
    if not isinstance(info, str):
        info = json.dumps(info, ensure_ascii=False)
    with open(new_file, 'a', encoding='utf-8') as fin:
        fin.write(info + '\n')

def judge_answer(question, url, ref_answer, answer):
    prompt = judge_prompt.format(question = question, url=url, target = ref_answer, predicted_answer = answer)
    messages = [{"role": "system", "content": "你是一个智能助手，请根据给定图片、问题、标准答案和模型预测的答案来评估模型的回答是否正确。"}]
    messages.append({"role": "user", "content": prompt})
    output,_ = call_model(messages, "gpt-4o-0806")
    correct = "C"
    try:
        match = re.search(r"(A|B|C)", output)
        correct = match.group(0) if match else "C" 
    except:
        output = "error"
    judge = {"judge": output.strip(), "score": correct}
    return judge

def process_line(line):
    score1 = line.get("score1", "")
    score2 = line.get("score2", "")
    if score1 != "" and score2 != "":
        write_to_file(line)
        return 1
    
    system_prompt = "你是一个智能助手。"

    question1 = line['recognition_question']
    ref_answer1 = line['recognition_answer']
    url = line["image_url"]
    prompt1 = f"图片：{url}\n问题：{question1}"
    messages1 = [{"role": "system", "content": system_prompt}]
    messages1.append({"role": "user", "content": prompt1})

    question2 = line['final_question']
    ref_answer2 = line['final_answer']
    prompt2 = f"图片：{url}\n问题：{question2}"
    messages2 = [{"role": "system", "content": system_prompt}]
    messages2.append({"role": "user", "content": prompt2})

    try:
        output1 = line.get("model_output1", "")  
        line['model_output1'] = output1

        output2 = line.get("model_output2", "") 
        line['model_output2'] = output2

        if output1 == "" or output2 == "":
            line['info'] = "模型输出为空"
            write_to_file(line)
            return 0
        
        judge1 = judge_answer(question1, url, ref_answer1, output1)
        if judge1['judge'] == "" or judge1['judge'] == "error":
            line['info'] = "评判出错"
            write_to_file(line)
            return 0

        judge2 = judge_answer(question2, url, ref_answer2, output2)
        if judge2['judge'] == "" or judge2['judge'] == "error":
            line['info'] = "评判出错"
            write_to_file(line)
            return 0

        line['judge1'] = judge1
        line['score1'] = judge1['score']

        line['judge2'] = judge2
        line['score2'] = judge2['score']
        write_to_file(line)
        return 1
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        write_to_file(line)
        return 0

def calculate_accuracies(group):
    total_questions = len(group)
    total_correct1 = group[group['score1'] == "A"].shape[0]
    total_incorrect1 = group[group['score1'] == "B"].shape[0]
    total_not_attempted1 = group[group['score1'] == "C"].shape[0]

    total_correct2 = group[group['score2'] == "A"].shape[0]
    total_incorrect2 = group[group['score2'] == "B"].shape[0]
    total_not_attempted2 = group[group['score2'] == "C"].shape[0]
    
    total_correct_accuracy1 = total_correct1 / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy1 = total_incorrect1 / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy1 = total_not_attempted1 / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy1 = total_correct1 / (total_correct1 + total_incorrect1) if (total_correct1 + total_incorrect1) > 0 else 0
    
    f1_1 = 2 * total_given_attempted_accuracy1 * total_correct_accuracy1 / (total_given_attempted_accuracy1+ total_correct_accuracy1) if (total_given_attempted_accuracy1+ total_correct_accuracy1) > 0 else 0

    total_correct_accuracy2 = total_correct2 / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy2 = total_incorrect2 / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy2 = total_not_attempted2 / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy2 = total_correct2 / (total_correct2 + total_incorrect2) if (total_correct2 + total_incorrect2) > 0 else 0
    
    f1_2 = 2 * total_given_attempted_accuracy2 * total_correct_accuracy2 / (total_given_attempted_accuracy2+ total_correct_accuracy2) if (total_given_attempted_accuracy2+ total_correct_accuracy2) > 0 else 0

    return pd.Series({'correct_recognition': total_correct_accuracy1, 'incorrect_recognition': total_incorrect_accuracy1, 'not_attempted_recognition': total_not_attempted_accuracy1, "given_attempted_accuracy_recognition": total_given_attempted_accuracy1, "F1_recognition": f1_1,
    'correct_final': total_correct_accuracy2, 'incorrect_final': total_incorrect_accuracy2, 'not_attempted_final': total_not_attempted_accuracy2, "given_attempted_accuracy_final": total_given_attempted_accuracy2, "F1_final": f1_2})

# ==============================

print(f"Model Name: {call_modelname}")

origin_file = "data/chinese_simplevqa.jsonl"
folder = "evaluation/models_results"
if not os.path.exists(folder):
    os.makedirs(folder)
new_file = f"{folder}/simplevqa_{call_modelname}.jsonl"
# ==============================

if __name__ == "__main__":
    import time
    start_time = time.perf_counter()   
    data_new = []
    with open(origin_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            data_new.append(data)
    fin =  open(new_file, "w", encoding='utf-8')

    with Pool(processes=10) as pool:
        sleep_time = random.uniform(0, 1)
        sleep(sleep_time)
        results = list(tqdm(pool.imap(process_line, data_new), total=len(lines)))       
        correct = np.sum(np.array(results))
        print("成功数： ", correct)   
    k = 0 
    while correct < 1100 and k < 3:
        k += 1
        print("失败数量过多，重试")
        start_time = time.perf_counter()    
        origin_file = new_file
        with open(origin_file, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [json.loads(line) for line in lines]
        new_file = f"{new_file}_{k}.jsonl"
        fin =  open(new_file, "w", encoding='utf-8')
        with Pool(processes=1) as pool:
            results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))       
            correct = np.sum(np.array(results))
            print("成功数： ", correct)   
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) / 60
    print(f"执行耗时: {execution_time_ms} mins")
    with open(new_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        datas = [json.loads(line) for line in lines]
        df = pd.json_normalize(datas)
        
        # 提取“人工分类”字段的主要类别（'|'之前的部分）
        df['主类别'] = df['Topic'].apply(lambda x: x.split('|')[0] if pd.notna(x) else x)
        
        # 对每个主类别分组计算精度
        accuracy_by_category = df.groupby('主类别').apply(calculate_accuracies).reset_index()

        # 计算总体精度
        overall_accuracy = calculate_accuracies(df)
        overall_accuracy['主类别'] = 'Overall'
        overall_accuracy = overall_accuracy.to_frame().T
        
        # 合并总体精度和按主类别计算的精度
        final_df = pd.concat([overall_accuracy, accuracy_by_category], ignore_index=True)
        
        # 保存结果到CSV文件
        output_file = new_file2.replace(".jsonl", ".csv")
        final_df.to_csv(output_file, index=False)
        print(f"Accuracy results saved to {output_file}")