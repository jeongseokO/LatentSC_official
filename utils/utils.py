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
    - int: The selected response index (e.g., 0 or 2).

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
    Searches the GPT response (answer_text) for a line containing '#### True' or '#### False'
    and returns True or False. Returns 'No Match' if no match is found.
    """
    # Search for '#### True' or '#### False' pattern
    m = re.search(r'####\s*(True|False)\b', answer_text)
    if not m:
        return "No Match"
    return True if m.group(1) == 'True' else False

def extract_boolean(response_str):
    """
    Extracts 'yes' or 'no' (case-insensitive) from the given string and returns a boolean.
    Returns 'No Match' if neither is found.
    """
    match = re.search(r'\b(yes|no)\b', response_str, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'yes'
    else:
        return "No Match"

def extract_alphabet(text):
    """
    Captures a single uppercase letter A–Z following '####',
    allowing optional spaces or braces.
    """
    pattern = r'#{4}\s*\{?([A-Z])\}?'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        if answer in string.ascii_uppercase:
            return answer
    return "No Match"

ANS_RE = re.compile(r"#### (\$?\-?[0-9\.\,]+)")
INVALID_ANS = "No Match"

def extract_number(completion):
    """
    Finds all numeric matches in the string and returns the last one,
    cleaned of commas and dollar signs. Returns 'No Match' if none found.
    """
    # Find all numeric patterns
    matches = ANS_RE.findall(completion)
    if matches:
        match_str = matches[-1].strip()
        match_str = match_str.replace(",", "").replace("$", "")
        return match_str
    else:
        return INVALID_ANS

def extract_string(text):
    """
    Uses a regular expression to find a numeric or alphabetic token following '####'
    and returns it. Returns 'No Match' if no match is found.
    """
    pattern = r'#### ([\w\\{}]+)'
    # Search for the pattern
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return "No Match"

def extract_triviaqa(text):
    """
    Uses a regular expression to capture any text following '####'
    and returns it without trailing periods. Returns 'No Match' if none found.
    """
    pattern = r'#{4}\s*(.*)'
    # Search for the pattern
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().replace(".", "")
    else:
        return "No Match"

def safe_convert_to_float(value):
    """
    Attempts to convert a cleaned string to float.
    Non-convertible strings are returned unchanged.
    """
    value = value.replace("\\", "")
    try:
        return float(value)
    except ValueError:
        return value

def remove_boxed(s):
    """
    Removes '\\boxed{...}' wrapper from a LaTeX string.
    Returns the inner content or 'No Match' if format is invalid.
    """
    left = "\\boxed{"
    try:
        assert s.startswith(left) and s.endswith("}")
        return s[len(left):-1]
    except AssertionError:
        return "No Match"

def last_boxed_only_string(string):
    """
    Extracts the content of the last '\\boxed{...}' or '\\fbox{...}' in the string.
    Returns inner content cleaned of dollar signs and spaces or 'No Match' if not found.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return "No Match"

    # Locate the closing brace
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return "No Match"
    retval = remove_boxed(string[idx:right_brace_idx + 1])
    return retval.replace("$", "").replace(" ", "")

def last_boxed_only_string_for_triviaqa(string):
    """
    Extracts the content of the last '\\boxed{...}' or '\\fbox{...}' (including formatting),
    or returns 'No Match' if none found.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return "No Match"

    # Locate the closing brace
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return "No Match"
    return remove_boxed(string[idx:right_brace_idx + 1])

def majority_vote(ans_list):
    """
    Returns the most common answer from the list.
    If 'No Match' is most common and other answers exist,
    returns the second most common answer.
    """
    # Count frequencies
    count = Counter(ans_list)
    most_common = count.most_common(1)[0][0]
    # If 'No Match' is the top, pick the next most common
    if most_common == "No Match" and len(count) > 1:
        most_common = count.most_common(2)[1][0]
    return most_common

def majority_vote_with_conf(ans_list):
    """
    Returns the most common answer, its confidence (frequency ratio),
    and the indices where it occurs.
    """
    count = Counter(ans_list)
    most_common, most_count = count.most_common(1)[0]
    # If 'No Match' is top, choose the runner-up
    if most_common == "No Match" and len(count) > 1:
        most_common, most_count = count.most_common(2)[1]
    # Calculate confidence as ratio of occurrences
    confidence = most_count / len(ans_list)
    # Identify all indices of the most common answer
    indices = [i for i, ans in enumerate(ans_list) if ans == most_common]
    return most_common, confidence, indices

def calculate_entropy(logprobs):
    """
    Calculates the entropy for each token's log probability distribution.
    """
    entropy_list = []
    for token_logprobs in logprobs:
        # Convert log probabilities to probabilities
        probabilities = [
            math.exp(lp.logprob)
            for lp in token_logprobs.values()
            if lp.logprob != -float('inf')
        ]
        # Compute entropy: -Σ(p * log(p))
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        entropy_list.append(entropy)
    return entropy_list

def normalize_answer(s):
    """
    Lowercases text and removes punctuation, articles, underscores, and extra whitespace.
    """
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

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()
