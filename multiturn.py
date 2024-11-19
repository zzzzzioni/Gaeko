from openai import AzureOpenAI
import yaml

##### load GPT #####
with open('openai_gpt.yaml') as f:
    config = yaml.safe_load(f)

api_version = config["config"]["api_version"]
azure_endpoint = config["config"]["azure_endpoint"]
api_key = config["config"]["api_key"]

client = AzureOpenAI(
    api_version = api_version,
    azure_endpoint = azure_endpoint,
    api_key = api_key
)  

##### load prompt #####
with open('classification.txt', encoding='utf-8') as file:
    clssf_prompt =  file.read()
    
with open('history.txt', encoding='utf-8') as file:
    hist_prompt =  file.read()

##### set user conversation state #####
conversation_state = {
    "collected_group": [],  # 수집된 그룹 정보
    "collected_history" : [],  # 수집된 이력 정보 
    "pending_question": None,  # 추가 질문
    "task_completed": False  # 분류 완료 여부
}

#################################################################
#                            취향 조사                            #
#################################################################
def generate_follow_up_question():
    ### 추가 질문을 위한 함수 
    global conversation_state
    print("""Gaeko🦎 : 네가 좋아하는 향에 대해 조금 더 알면, 네게 정말 딱 맞는 향수를 추천해줄 수 있을 것 같아!
일상생활을 하면서 네가 좋다고 느꼈던 아무 냄새나 다 말해줘!🤩
심지어 지하주차장 냄새, 커피향, 귤냄새 이런 것도 괜찮아😚""")
    
    user_input = input("User🧸 : ")

    response = client.chat.completions.create(
        model="hatcheryOpenaiCanadaGPT4o",
        messages=[
            {"role": "system", "content": clssf_prompt},  # 위의 system 메시지 그대로
            {"role": "user", "content": f"<input>-'사용자'의 입력 : {user_input}</input>"}
        ]
    )
    extracted_groups = response.choices[0].message.content.strip("[]").replace("'", "").split(", ")

    # 2. 수집된 데이터 저장
    conversation_state["collected_group"].extend(extracted_groups)
    
    # conversation_state["task_completed"] = True
    # return classify_user_preference()


def classify_user_preference():
    # 수집된 데이터 기반으로 결과 출력
    unique_groups = list(set(conversation_state["collected_group"]))
    return print(f"User🧸가 좋아하는 향 그룹: {unique_groups}")

def process_user_group(user_input):
    global conversation_state
    ### 첫 질문을 위한 함수 

    # 1. 현재 입력으로 그룹 추출
    response = client.chat.completions.create(
        model="hatcheryOpenaiCanadaGPT4o",
        messages=[
            {"role": "system", "content": clssf_prompt},  # 위의 system 메시지 그대로
            {"role": "user", "content": f"<input>-'사용자'의 입력 : {user_input}</input>"}
        ]
    )
    extracted_groups = response.choices[0].message.content.strip("[]").replace("'", "").split(", ")

    # 2. 수집된 데이터 저장
    conversation_state["collected_group"].extend(extracted_groups)

    # 3. 부족한 정보 판단
    if len(set(conversation_state["collected_group"])) < 3:  # 최소 3개 그룹 필요
        conversation_state["pending_question"] = generate_follow_up_question()
        return conversation_state["pending_question"]
    else:
        conversation_state["task_completed"] = True
        return classify_user_preference()

##### first chat #####
print("""
Gaeko🦎 :
안녕!🦎 난 향수 전문가 '개코'라고 해! 킁킁🦎 네가 딱 좋아할 향수를 추천해주기 위해 몇가지 간단한 질문을 할 거야.
먼저, 넌 어떤 냄새를 좋아해?🤔
나는 포근한 살냄새와 찻잎냄새를 좋아해, 좀 더 전문적?으로 이야기하면 머스크 계열과 화이트 플로럴 향을 좋아하지 ㅎㅎ
나처럼, 네가 좋아하는 향을 자유롭게 말해줘!🦎
        """)

user_input = input("User🧸 : ")

process_user_group(user_input)

#################################################################
#                            과거 이력                            #
#################################################################

def generate_follow_up_question_hist():
    ### 추가 질문을 위한 함수 
    global conversation_state
    print("""Gaeko🦎 : 
네 취향에 대한 구체적인 정보가 조금 더 필요해!🥹
혹시 네가 좋아하는 구체적인 향수나 브랜드에 대해 조금 더 알려줄 수 있을까?
그렇다면 내가 너에게 더 정확한 추천을 해줄 수 있을 거야! 🤓
""")
    
    user_input = input("User🧸 : ")

    response = client.chat.completions.create(
        model="hatcheryOpenaiCanadaGPT4o",
        messages=[
            {"role": "system", "content": hist_prompt},  # 위의 system 메시지 그대로
            {"role": "user", "content": f"<input>-'사용자'의 입력 : {user_input}</input>"}
        ]
    )
    extracted_groups = response.choices[0].message.content.strip("[]").replace("'", "").split(", ")

    # 2. 수집된 데이터 저장
    conversation_state["collected_history"].append(extracted_groups)
    
    conversation_state["task_completed"] = True
    return classify_user_preference(), classify_user_history()

def classify_user_history():
    # 수집된 데이터 기반으로 결과 출력
    unique_groups = list(conversation_state["collected_history"])
    return print(f"User🧸의 이력: {unique_groups}")


def process_user_input(user_input):
    global conversation_state
    # initialize state
    conversation_state["pending_question"] = None 
    conversation_state["task_completed"] = False
    
    ### 첫 질문을 위한 함수 

    # 1. 현재 입력으로 그룹 추출
    response = client.chat.completions.create(
        model="hatcheryOpenaiCanadaGPT4o",
        messages=[
            {"role": "system", "content": hist_prompt},  # 위의 system 메시지 그대로
            {"role": "user", "content": f"<input>-'사용자'의 입력 : {user_input}</input>"}
        ]
    )
    extracted_groups = response.choices[0].message.content.strip("[]").replace("'", "").split(", ")

    # 2. 수집된 데이터 저장
    conversation_state["collected_history"].append(extracted_groups)

    # 3. 부족한 정보 판단
    if len(conversation_state["collected_history"]) < 2:  # 최소 3개 그룹 필요
        conversation_state["pending_question"] = generate_follow_up_question_hist()
        return conversation_state["pending_question"]
    else:
        conversation_state["task_completed"] = True
        return classify_user_preference(), classify_user_history()
    

print("""
Gaeko🦎 :
좋았어! 네가 좋아하는 향이 어떤 느낌인지 알 것 같아 🧐
자세히 말해줘서 고마워🦎
""")

print("""
Gaeko🦎 :
그럼 이제 다음 질문으로 넘어갈게!🦎 이번엔 조금 더 구체적으로 물어볼거야.
혹시 좋아하는 향수 브랜드나 특정 향수가 있어?
만약 특별히 없다면, 예전에 네가 사용했던 향수를 말해줘! 
향수 이름이나 브랜드명이 정확하게 기억이 안 나면, 그냥 대강 생각나는대로 말해줘도 돼!😋
음, 내 경우에는 샤넬 레젝 1957이랑 톰포트 화이트스웨이드를 특별히 좋아하고, 딥디크 브랜드 향수는 다 좋아 ㅎㅎ
넌 어때?
""")

user_input = input("User🧸 : ")

process_user_input(user_input)

print("""
Gaeko🦎 :
정말 고마워! 너에 대해 잔뜩 알게 되었으니, 이제 네가 좋아할만한 향수를 추천해줄게 😎
""")