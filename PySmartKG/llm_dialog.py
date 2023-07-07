import requests
import json
import openai
from kg_engine import generate_response_message


def generate_prompt(query, trace_logs):
    prompt_template_path = "data/dialog_prompt.txt"

    # 从文件中读取 prompt 模板
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()
        # 将 trace_logs 列表转换为一个长字符串
        trace_logs_str = '\n'.join(trace_logs)
        prompt = prompt_template.replace('{related_information}', trace_logs_str).replace('{query_str}', query)
        return prompt

    return query


def gpt_openai_process(prompt):
    with open('data/openai_key.txt', 'r') as file:
        openai.api_key = file.read().strip()

    openai.api_type = 'azure'
    openai.api_base = "https://avidemo0.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    response = openai.ChatCompletion.create(
            engine="gpt40314",
            # replace this value with the deployment name you chose when you deployed the associated model.
            messages=[{"role": "user","content": prompt}],
            temperature=0,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)

    text = response.choices[0].message.content.strip()

    return text

def chatglm_process(prompt):
    """
    :return:
    """
    url = f"http://192.168.50.189:7087/api/chat"
    my_data = {"text": prompt}
    headers = {'content-type': 'application/json'}
    # 提交form格式数据
    data = {"data": my_data}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    assert r.status_code == 200, f"返回的status code不是200，请检查"
    res = r.json()
    print(json.dumps(res, indent=4, ensure_ascii=False))
    response = res["response"]
    return response

def get_response_from_llm(query, final_entities, trace_logs):

    try:
        prompt = generate_prompt(query, trace_logs)
        resp_text = chatglm_process(prompt)  # 更改成chatglm
        resp_message = f"{resp_text}"
    except Exception as e:
        resp_message = generate_response_message(final_entities, trace_logs)#f"在调用大模型的过程中出错"

    return resp_message