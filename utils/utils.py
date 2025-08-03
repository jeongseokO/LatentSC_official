import re
from collections import Counter
import math
import string
def extract_response_id(answer: str, valid_options: list[int]) -> int:
    """
    Extracts the first valid integer from the answer that matches one of the expected options.

    Parameters:
    - answer (str): The raw output string from the LLM.
    - valid_options (list[int]): The list of valid indices, e.g., [0, 2].

    Returns:
    - int: The selected response index (e.g., 0 or 2)

    Raises:
    - ValueError: If no valid number is found in the answer.
    """
    # Find all integer numbers in the answer
    numbers = re.findall(r'\b\d+\b', answer)
    for num_str in numbers:
        num = int(num_str)
        if num in valid_options:
            return num
    print(f"No valid response ID found in answer: {answer}")
    return -1
def extract_boolean_from_answer(answer_text: str) -> bool:
    """
    GPT 응답(answer_text)에서 '#### True' 또는 '#### False' 라인을 찾아
    True/False로 리턴합니다. 못 찾으면 None을 리턴합니다.
    """
    # '#### True' 또는 '#### False' 패턴 검색
    m = re.search(r'####\s*(True|False)\b', answer_text)
    if not m:
        return "No Match"
    return True if m.group(1) == 'True' else False
def extract_boolean(response_str):
    """
    주어진 문자열에서 'True' 또는 'False'라는 단어를 추출하여 Boolean으로 반환합니다.
    만약 두 단어 모두 없으면 ValueError를 발생시킵니다.
    """
    match = re.search(r'\b(yes|no)\b', response_str, re.IGNORECASE)
    if match:
        # 매칭된 단어를 소문자로 변환한 후, 'true'면 True, 아니면 False 반환
        return match.group(1).lower() == 'yes'
    else:
        return "No Match"
def extract_alphabet(text):
    # #### 뒤에 공백이나 중괄호가 있어도 A~Z 한 글자를 캡처
    pattern = r'#{4}\s*\{?([A-Z])\}?'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        # A~Z 전부 허용
        if answer in string.ascii_uppercase:
            return answer
    return "No Match"

ANS_RE = re.compile(r"#### (\$?\-?[0-9\.\,]+)")
INVALID_ANS = "No Match"

def extract_number(completion):
    # 전체 문자열에서 모든 매치를 찾고 마지막 값을 반환합니다.
    matches = ANS_RE.findall(completion)
    if matches:
        match_str = matches[-1].strip()  # 마지막 매치를 사용
        match_str = match_str.replace(",", "").replace("$", "")
        return match_str
    else:
        return INVALID_ANS
def extract_string(text):
    # #### 뒤에 오는 숫자 또는 문자를 찾는 정규 표현식
    pattern = r'#### ([\w\\{}]+)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 결과를 반환
    if match:
        return match.group(1)
    else:
        return "No Match"
def extract_triviaqa(text):
    # #### 뒤에 오는 텍스트를 찾는 정규 표현식
    pattern = r'#{4}\s*(.*)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 텍스트를 반환
    if match:
        return match.group(1).strip().replace(".", "")
    else:
        return "No Match"
def safe_convert_to_float(value):
    value = value.replace("\\", "")
    try:
        # 문자열을 정수로 변환 시도
        return float(value)
    except ValueError:
        # 변환할 수 없는 경우 원래 값을 반환
        return value
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return "No Match"
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return "No Match"

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = "No Match"
    else:
        retval = remove_boxed(string[idx:right_brace_idx + 1]).replace("$", "").replace(" ", "")
    
    return retval
def last_boxed_only_string_for_triviaqa(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return "No Match"

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = "No Match"
    else:
        retval = remove_boxed(string[idx:right_brace_idx + 1])

    return retval
def majority_vote(ans_list):
    # Majority vote
    count = Counter(ans_list)
    # 가장 많이 나온 답과 그 개수 찾기
    most_common = count.most_common(1)[0][0]
    # most_count = count.most_common(1)[0][1]

    # "Not a Number"가 가장 많이 나온 경우 두 번째로 많이 나온 답 찾기
    if most_common == "No Match" and len(count) > 1:
        most_common = count.most_common(2)[1][0]
        # most_count = count.most_common(2)[1][1]
    return most_common

def majority_vote_with_conf(ans_list):
    # Majority vote
    count = Counter(ans_list)
    # 가장 많이 나온 답과 그 개수 찾기
    most_common = count.most_common(1)[0][0]
    most_count = count.most_common(1)[0][1]

    # "No Match"가 가장 많이 나온 경우 두 번째로 많이 나온 답 찾기
    if most_common == "No Match" and len(count) > 1:
        most_common = count.most_common(2)[1][0]
        most_count = count.most_common(2)[1][1]
    
    # Confidence 계산
    confidence = most_count / len(ans_list)

    # 가장 자주 나온 답의 인덱스 추출
    indices = [i for i, ans in enumerate(ans_list) if ans == most_common]

    return most_common, confidence, indices

def extract_alphabet(text):
    pattern = r'#{4}\s*([A-Z])'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        if answer in ["A", "B", "C", "D", "E"]:
            return answer
    return "No Match"

def extract_string(text):
    # #### 뒤에 오는 숫자 또는 문자를 찾는 정규 표현식
    pattern = r'#### ([\w\\{}]+)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 결과를 반환
    if match:
        return match.group(1)
    else:
        return "No Match"
def extract_triviaqa(text):
    # #### 뒤에 오는 텍스트를 찾는 정규 표현식
    pattern = r'#{4}\s*(.*)'

    # 정규 표현식과 일치하는 부분을 찾기
    match = re.search(pattern, text)
    
    # 일치하는 부분이 있다면, 텍스트를 반환
    if match:
        return match.group(1).strip().replace(".", "")
    else:
        return "No Match"


def calculate_entropy(logprobs):
    """각 토큰의 엔트로피를 계산합니다."""
    entropy_list = []
    for token_logprobs in logprobs:
        # Log probability 값에서 확률로 변환
        probabilities = [math.exp(lp.logprob) for lp in token_logprobs.values() if lp.logprob != -float('inf')]
        
        # 엔트로피 계산: -Σ(p * log(p))
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        entropy_list.append(entropy)
    
    return entropy_list

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()