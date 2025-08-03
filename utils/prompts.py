import string

def get_messages(context=None, question=None, cot_ex=None, choices=None, model_name=None, dataset=None, response=None, completions=None, emphasize=False, format=True):      
    if dataset == "MMLU":
        question = f"{question}\n\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}"
        messages = [
            {"role": "system", "content": "You are a methodical problem solver, adept at solving complex multiple choice problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question'] + "\nFinish with exactly one sentence:'\n\nTherefore, the answer is #### {Letter}'"})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + "\nFinish with exactly one sentence:'\n\nTherefore, the answer is #### {Letter}'"})
    
    elif dataset in ["truthfulqa_mcqa", "commonsense_qa"]:
        if choices:
            # Generate alphabet labels for choices
            labels = list(string.ascii_uppercase)
            # Mapping choices to labels
            option_lines = "\n".join(
                f"{labels[i]}: {opt}"
                for i, opt in enumerate(choices)
                if i < len(labels)
            )
            # Attach the question with choices
            question_with_choices = f"{question}\n\n{option_lines}"
        else:
            # If no choices are provided, use the question as is
            question_with_choices = question

        # system message
        messages = [
            {"role": "system", "content": (
                "You are a methodical problem solver, adept at solving complex multiple choice problems. "
                "Conclude your explanation with the answer in a '#### {Alphabet answer}' format, "
                "where the answer is solely an alphabet."
            )}
        ]

        # few-shot CoT Examples
        for key in cot_ex:
            messages.append({
                "role": "user",
                "content": cot_ex[key]['Question'] + "\nConclude your explanation with the answer in a '#### {Letter}'"})
            
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})

        # user question
        messages.append({
            "role": "user",
            "content": question_with_choices
            + "\nConclude your explanation with the answer in a '#### {Letter}'"})
        
    elif dataset == "mmlu_short":
        messages = [
            {"role": "system", "content": "You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {answer}' format"},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
        
        messages.append({"role": "user", "content": "Solve the following problem: " + question + "\nEnsure your final answer is presented within the format '#### {answer}'."})
    
    elif dataset == "race" and context: 
        messages = [
        {"role": "system", "content": "You are a methodical problem solver, adept at solving complex problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet"},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
        
        messages.append({"role": "user", "content": context + "\nQuestion: : " + question + "\nOptions: A:"+choices[0]+", B:"+choices[1]+ ", C:"+choices[2]+", D:"+choices[3]  + "Ensure your final answer is presented within the format '#### {Alphabet answer}'."})
    
    elif dataset == "aqua":
        messages = [
        {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {Alphabet answer}' format, where the answer is solely an alphabet"},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": "Example Problem " + key + ": " + cot_ex[key]})
        
        messages.append({"role": "user", "content": "Solve the following mathematical problem: " + question + "\nOptions: "+choices[0]+", "+choices[1]+ ", "+choices[2]+", "+choices[3]+", "+choices[4]+" Ensure your final answer is presented within the format '#### {Alphabet answer}'."})
    
    elif dataset == "GSM8K":
        if format:
            messages = [
            {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number."},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": cot_ex[key]['Question'] + "Ensure your final answer is presented within the format '#### {numeric answer}'."})
                messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
            messages.append({"role": "user", "content": question + " Ensure your final answer is presented within the format '#### {numeric answer}'."})
        else:
            messages = [
            {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer"},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": cot_ex[key]['Question']})
                messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
            messages.append({"role": "user", "content": question})
    elif dataset == "MATH":
        messages = [
        {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '$\\boxed{answer}$' format."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question'] + " Conclude your explanation with the answer in a '$\\boxed{answer}$' format."})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + " Conclude your explanation with the answer in a '$\\boxed{answer}$' format."})
    
    elif dataset in ["triviaqa","nq"]:
        messages = [
        {"role": "system", "content": "You are a methodical problem solver, adept at answering questions. Conclude your explanation with the answer in a '\\boxed{answer}' format."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question'] + "Ensure your final answer is presented within the format '\\boxed{answer}'. Let's think step by step."})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        
        messages.append({"role": "user", "content": question  + "Ensure your final answer is presented within the format '\\boxed{answer}'. Let's think step by step."})
    
    elif dataset in ["truthfulqa"]:
        if emphasize:
            messages = [
            {"role": "system", "content": (
                "You are a methodical problem solver. Respond in one or two sentences only, and conclude your explanation with the final answer."
            )},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": cot_ex[key]['Question'] + " Let's think step by step and conclude your explanation with the final answer."})
                messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
            messages.append({"role": "user", "content": question + " Let's think step by step and conclude your explanation with the final answer."})
        else:
            messages = [
            {"role": "system", "content": "You are a methodical problem solver. Respond concisely in one or two sentences only."},
            ]
            for key in cot_ex:
                messages.append({"role": "user", "content": cot_ex[key]['Question']})
                messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
            messages.append({"role": "user", "content": question})

    elif dataset in ["cnn_dailymail"]:
        messages = [
        {"role": "system", "content": "You are a helpful assistant, adept at summarizing long pieces of text."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": "This is a long piece of text that you want to summarize. It contains multiple sentences and paragraphs. The goal is to generate a concise summary that captures the main points of the text. Think step by step.\nArticle: " + cot_ex[key]['Question']})
            messages.append({"role": "assistant", "content": "Highlights: " + cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": "This is a long piece of text that you want to summarize. It contains multiple sentences and paragraphs. The goal is to generate a concise summary that captures the main points of the text. Think step by step.\nArticle: " + question})
    elif dataset in ["usc"]:
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        ]
        messages.append({"role": "user", "content": "Here are multiple reasoning paths for a task. Select the most consistent and plausible path based on consensus:\n\n"})
        for idx, path in enumerate(completions):
            messages[-1]['content'] += f"Path {idx + 1}: {path}\n\n"
        messages[-1]['content'] += "Which path is the most consistent? Conclude your explanation with the answer in a 'Path {number}' format."
        
    return messages

def get_cot_Ex(dataset):
    cot_ex = {}
    if dataset == "MMLU":
        cot_ex['example1'] = {
                "Question": "What is the value of p in 24 = 2p?\n\nA: p=4\nB: p=8\nC: p=12\nD: p=24",
                "Answer": "Step 1: To find the value of p in the equation 24 = 2p, we start by identifying the equation given. Step 2: Next, we need to isolate p by dividing both sides of the equation by 2. Step 3: This gives us 24 ÷ 2 = 2p ÷ 2, simplifying to 12 = p. Step 4: Therefore, the value of p is 12. Therefore, the answer is #### C"
            }
        cot_ex['example2'] = {
                "Question": "What is the remainder of 21 divided by 7?\n\nA: 21\nB: 7\nC: 1\nD: None of these",
                "Answer": "Step 1: To find the remainder when 21 is divided by 7, we begin by performing the division. Step 2: 21 divided by 7 equals 3, which is an integer. Step 3: Since the result is an integer, there is no remainder. Step 4: Therefore, when 21 is divided by 7, the remainder is 0. Step 5: The answer is #### D."
            }
    if dataset in ["truthfulqa_mcqa", "commonsense_qa"]:
        cot_ex['example1'] = {
                "Question": "What is the value of p in 24 = 2p?\n\nA: p=4\nB: p=8\nC: p=12\nD: p=24",
                "Answer": "Step 1: To find the value of p in the equation 24 = 2p, we start by identifying the equation given. Step 2: Next, we need to isolate p by dividing both sides of the equation by 2. Step 3: This gives us 24 ÷ 2 = 2p ÷ 2, simplifying to 12 = p. Step 4: Therefore, the value of p is 12. Therefore, the answer is #### C"
            }
        cot_ex['example2'] = {
                "Question": "What is the remainder of 21 divided by 7?\n\nA: 21\nB: 7\nC: 1\nD: None of these",
                "Answer": "Step 1: To find the remainder when 21 is divided by 7, we begin by performing the division. Step 2: 21 divided by 7 equals 3, which is an integer. Step 3: Since the result is an integer, there is no remainder. Step 4: Therefore, when 21 is divided by 7, the remainder is 0. Step 5: The answer is #### D."
            }
    elif dataset == "GSM8K":
        cot_ex['example1'] = {
            "Question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "Answer": "Step 1: We start with 15 trees. Step 2: Later we have 21 trees. Step 3: The difference must be the number of trees they planted. Step 4: So, they must have planted 21 - 15 = 6 trees. The answer is #### 6."
        }
        cot_ex['example2'] = {
            "Question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "Answer": "Step 1: There are 3 cars in the parking lot already. Step 2: 2 more arrive. Step 3: Now there are 3 + 2 = 5 cars. The answer is #### 5."
        } 
    elif dataset == "MATH":
        cot_ex['example1'] = {
            "Question": "Evaluate $\\lfloor 14.6 \\rfloor - \\lceil -14.6 \\rceil$.\n",
            "Answer": "Step 1: The greatest integer less than $14.6$ is $14$. Step 2: The smallest integer greater than $-14.6$ is $-14$. Step 3: Therefore, the equation can be rewritten as $14 - (-14)$, or $14 + 14$. The answer is $\\boxed{28}$."
        }
        cot_ex['example2'] = {
            "Question": "What is the sum of all numbers $a$ for which the graph of $y = x^2 + a$ and the graph of $y = ax$ intersect one time?",
            "Answer": "Step 1: If these two graphs intersect, the point of intersection occurs when $x^2 + a = ax$. Step 2: This can be rewritten as $x^2 - ax + a = 0$. Step 3: This quadratic has one solution exactly when the discriminant is equal to zero: $(-a)^2 - 4 \\cdot 1 \\cdot a = 0$. Step 4: This simplifies to $a(a - 4) = 0$. Step 5: There are exactly two values of $a$ for which the line and parabola intersect one time, namely $a = 0$ and $a = 4$. Step 6: The sum of these values is $0 + 4 = 4$. The answer is $\\boxed{4}$."
        }
    elif dataset == "triviaqa":
        cot_ex['example1'] = {
            "Question": "Where in England was Dame Judi Dench born?",
            "Answer": "To determine where Dame Judi Dench was born, we can follow these steps:\n1. Understand the query: We need to find the specific location in England where Dame Judi Dench was born.\n2. Recognize her fame: Dame Judi Dench is a renowned English actress known for her work in theater, television, and film.\n3. Historical context: Details about her birthplace are well-documented in her biographies and public records.\nAfter researching, Dame Judi Dench was born in York, England, on December 9, 1934. York is a historic city in the county of North Yorkshire. The answer is \\boxed{York, England}"
        }
        cot_ex['example2'] = {
            "Question": "From which country did Angola achieve independence in 1975?",
            "Answer": "To determine from which country Angola achieved independence in 1975, let’s analyze step by step:\n1. Understand Angola's colonial history: Angola was a colony for several centuries before gaining independence. Understanding its colonizing power will clarify this.\n2. Identify the European colonizing country: Angola was colonized by Portugal during the period of European exploration and colonization in Africa. Portuguese influence in Angola began in the late 15th century and solidified over the subsequent centuries.\n3. Independence movements in Africa: Many African nations gained independence during the mid-20th century. Angola's independence was part of a broader decolonization wave.\n4. Key events leading to independence: Angola’s path to independence included a prolonged struggle marked by an anti-colonial war (1961–1974). The Carnation Revolution in Portugal in 1974 significantly weakened Portuguese control over its colonies.\n5. Outcome: On November 11, 1975, Angola officially declared its independence from Portugal.\nThus, Angola achieved independence from Portugal in 1975. Therefore, the answer is \\boxed{Portugal}"
        }    
    elif dataset == "truthfulqa":
        cot_ex['example1'] = {
            "Question": "What is human life expectancy in the United States?",
            "Answer": (
                "Step 1: According to the latest data, life expectancy in the United States is approximately 77 years.\n"
                "Step 2: Recognize that this figure directly answers the question about life expectancy.\n"
                "Therefore, the answer is 77 years."
            )
        }

        cot_ex['example2'] = {
            "Question": "Who was president of the United States in 1955?",
            "Answer": (
                "Step 1: Recall that Dwight D. Eisenhower was elected president in 1952 and served two terms, covering the year 1955.\n"
                "Step 2: Conclude that Dwight D. Eisenhower was the incumbent president in 1955.\n"
                "Therefore, the answer is Dwight D. Eisenhower."
            )
        }

    elif dataset in ["cnn_dailymail"]:
        cot_ex['example1'] = {
            "Question": """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.""",
            "Answer": """Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday . Young actor says he has no plans to fritter his cash away . Radcliffe's earnings from first five Potter films have been held in trust fund ."""
        }
        cot_ex['example2'] = {
            "Question": """BOGOTA, Colombia (CNN) -- A key rebel commander and fugitive from a U.S. drug trafficking indictment was killed over the weekend in an air attack on a guerrilla encampment, the Colombian military said Monday. Alleged cocaine trafficker and FARC rebel Tomas Medina Caracas in an Interpol photo. Tomas Medina Caracas, known popularly as "El Negro Acacio," was a member of the high command of the Fuerzas Armadas Revolucionarias de Colombia and, according to Colombian and U.S. officials, helped manage the group's extensive cocaine trafficking network. He had been in the cross-hairs of the U.S. Justice Department since 2002. He was charged with conspiracy to import cocaine into the United States and manufacturing and distributing cocaine within Colombia to fund the FARC's 42-year insurgency against the government. U.S. officials alleged Medina Caracas managed the rebel group's sales of cocaine to international drug traffickers, who in turn smuggled it into the United States. He was also indicted in the United States along with two other FARC commanders in November 2002 on charges of conspiring to kidnap two U.S. oil workers from neighboring Venezuela in 1997 and holding one of them for nine months until a $1 million ransom was paid. Officials said the army's Rapid Response Force, backed by elements of the Colombian Air Force, tracked Medina Caracas down at a FARC camp in the jungle in the south of the country. "After a bombardment, the troops occupied the camp, and they've found 14 dead rebels so far, along with rifles, pistols, communications equipment and ... four GPS systems," Defense Minister Juan Manuel Santos said at a news conference. "The death of 'El Negro Acacio' was confirmed by various sources, including members of FARC itself." Medina Caracas commanded FARC's 16th Front in the southern departments of Vichada and Guainia. Established in 1964 as the military wing of the Colombian Communist Party, FARC is Colombia's oldest, largest, most capable and best-equipped Marxist rebel group, according to the U.S. Department of State. E-mail to a friend . Journalist Fernando Ramos contributed to this report.""",
            "Answer": """Tomas Medina Caracas was a fugitive from a U.S. drug trafficking indictment . "El Negro Acacio" allegedly helped manage extensive cocaine network . U.S. Justice Department indicted him in 2002 . Colombian military: He was killed in an attack on a guerrilla encampment ."""
        }

    return cot_ex

def get_messages_usc(question, responses, cot=False):
    """
    Generate messages for the USC (User-Specified Context) format.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    user_content = f"I have generated the following responses to the question: {question}"
    for idx, response in enumerate(responses):
        user_content += f"\nResponse {idx}: {response}"
    if cot == True:
        user_content += (
            "\nSelect the most consistent response based on majority consensus. "
            "Provide your final answer in the following format: #### {index number} (where index number is one of the responses, e.g., #### 0). Let's think step by step."
        )
    else:
        user_content += (
            "\nSelect the most consistent response based on majority consensus. "
            "Provide your answer in the following format: #### {index number} (where index number is one of the responses, e.g., #### 0)."
        )
    messages.append({"role": "user", "content": user_content})
    return messages



def get_messages_for_data(question, cot_ex, dataset):
    if dataset == "multiple_choice_cot":
        messages = [
            {"role": "system", "content": "Whenever you receive a question with answer choices, treat it as an open-ended problem—lay out your clear, step-by-step reasoning and a brief conclusion without ever mentioning the options, then conclude with exactly one sentence in this form:\n\nTherefore, the answer is #### {Letter}"},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question']})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + "\n\nTreat this like an open-ended problem: give me clear, step-by-step reasoning and a brief conclusion—without referring to the choice letters— then finish with exactly one sentence:'\n\nTherefore, the answer is #### {Letter}'"})
    elif dataset == "multiple_choice":
        messages = [
            {"role": "system", "content": "Whenever you receive a question with answer choices, treat it as an open-ended problem—lay out your brief conclusion without ever mentioning the options, then conclude with exactly one sentence in this form:\n\nTherefore, the answer is #### {Letter}"},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question']})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + "\n\nTreat this like an open-ended problem: give me a brief conclusion—without referring to the choice letters— then finish with exactly one sentence:'\n\nTherefore, the answer is #### {Letter}'"})
    
    elif dataset == "gsm8k":
        messages = [
        {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '$\\boxed{answer}$' format."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question'] + "Ensure your final answer is presented within the format '$\\boxed{answer}$'."})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + " Ensure your final answer is presented within the format '$\\boxed{answer}$'."})

    elif dataset == "math":
        messages = [
        {"role": "system", "content": "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '$\\boxed{answer}$' format."},
        ]
        for key in cot_ex:
            messages.append({"role": "user", "content": cot_ex[key]['Question'] + "Ensure your final answer is presented within the format '$\\boxed{answer}$'."})
            messages.append({"role": "assistant", "content": cot_ex[key]['Answer']})
        messages.append({"role": "user", "content": question + " Ensure your final answer is presented within the format '$\\boxed{answer}$'."})
    
    return messages

def get_cot_Ex_for_data(dataset=None, model_name=None, emphasize=False):
    cot_ex = {}
    if dataset == "multiple_choice_cot":
        cot_ex['example1'] = {
            "Question": (
                "What is the value of p in 24 = 2p?\n"
                "A: p=4\n"
                "B: p=8\n"
                "C: p=12\n"
                "D: p=24\n"
            ),
            "Answer": (
                "Step 1: Identify the given equation, which is 24 = 2p.\n"
                "Step 2: To isolate p, divide both sides by 2: 24 ÷ 2 = p.\n"
                "Step 3: Compute 24 ÷ 2 to get p = 12. Therefore, p equals to 12.\n\n"
                "Therefore, the answer is #### C."
            )
        }
        cot_ex['example2'] = {
            "Question": (
                "What is the least common multiple of 4 and 10?\n"
                "A: 14\n"
                "B: 20\n"
                "C: 40\n"
                "D: 60\n"
            ),
            "Answer": (
                "Step 1: To find the least common multiple (LCM) of 4 and 10, determine the prime factors of each number.\n"
                "Step 2: The prime factors of 4 are 2 and 2, which is 2².\n"
                "Step 3: The prime factors of 10 are 2 and 5.\n"
                "Step 4: Identify the highest power of each prime factor appearing in either number.\n"
                "Step 5: For 2 the highest power is 2²; for 5 it is 5.\n"
                "Step 6: Multiply these together: 2² × 5 = 4 × 5 = 20. Therefore, the least common multiple of 4 and 10 is 20.\n\n"
                "Therefore, the answer is #### B."
            )
        }

        cot_ex['example3'] = {
            "Question": (
                "Estimate 711 + 497. The sum is between which numbers?\n"
                "A: 50 and 400\n"
                "B: 450 and 700\n"
                "C: 750 and 1000\n"
                "D: 1,050 and 1,300\n"
            ),
            "Answer": (
                "To estimate 711 + 497, round each number to the nearest hundred—700 and 500. "
                "Adding those gives 1200, which lies between 1050 and 1300.\n\n"
                "Therefore, the answer is #### D."
            )
        }

    elif dataset == "multiple_choice":
        cot_ex['example1'] = {
            "Question": (
                "What is the value of p in 24 = 2p?\n"
                "A: p=4\n"
                "B: p=8\n"
                "C: p=12\n"
                "D: p=24\n"
            ),
            "Answer": (
                "Compute 24 ÷ 2 to get p = 12. Therefore, p equals to 12.\n\n"
                "Therefore, the answer is #### C."
            )
        }
        cot_ex['example2'] = {
            "Question": (
                "What is the least common multiple of 4 and 10?\n"
                "A: 14\n"
                "B: 20\n"
                "C: 40\n"
                "D: 60\n"
            ),
            "Answer": (
                "The least common multiple of 4 and 10 is 20.\n\n"
                "Therefore, the answer is #### B."
            )
        }

        cot_ex['example3'] = {
            "Question": (
                "Estimate 711 + 497. The sum is between which numbers?\n"
                "A: 50 and 400\n"
                "B: 450 and 700\n"
                "C: 750 and 1000\n"
                "D: 1,050 and 1,300\n"
            ),
            "Answer": (
                "It lies between 1050 and 1300.\n\n"
                "Therefore, the answer is #### D."
            )
        }
    elif dataset == "gsm8k":
        
        cot_ex['example1'] = {
            "Question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "Answer": "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is $\\boxed{6}$."
        }
        cot_ex['example2'] = {
            "Question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "Answer": "Step 1: There are 3 cars in the parking lot already. Step 2: 2 more arrive. Step 3: Now there are 3 + 2 = 5 cars. The answer is $\\boxed{5}$."
        }
    elif dataset == "math":
        cot_ex['example1'] = {
            "Question": "Evaluate $\\lfloor 14.6 \\rfloor - \\lceil -14.6 \\rceil$.\n",
            "Answer": "The greatest integer less than $14.6$ is $14$. The smallest integer greater than $-14.6$ is $-14$. Therefore, the equation can be rewritten as $14 - (-14)$, or $14 + 14$. The answer is $\\boxed{28}$."
        }
        cot_ex['example2'] = {
            "Question": "What is the sum of all numbers $a$ for which the graph of $y = x^2 + a$ and the graph of $y = ax$ intersect one time?",
            "Answer": "Step 1: If these two graphs intersect, the point of intersection occurs when $x^2 + a = ax$. Step 2: This can be rewritten as $x^2 - ax + a = 0$. Step 3: This quadratic has one solution exactly when the discriminant is equal to zero: $(-a)^2 - 4 \\cdot 1 \\cdot a = 0$. Step 4: This simplifies to $a(a - 4) = 0$. Step 5: There are exactly two values of $a$ for which the line and parabola intersect one time, namely $a = 0$ and $a = 4$. Step 6: The sum of these values is $0 + 4 = 4$. The answer is $\\boxed{4}$."
        }
    elif dataset == "triviaqa":
        cot_ex['example1'] = {
            "Question": "Where in England was Dame Judi Dench born?",
            "Answer": "To determine where Dame Judi Dench was born, we can follow these steps:\n1. Understand the query: We need to find the specific location in England where Dame Judi Dench was born.\n2. Recognize her fame: Dame Judi Dench is a renowned English actress known for her work in theater, television, and film.\n3. Historical context: Details about her birthplace are well-documented in her biographies and public records.\nAfter researching, Dame Judi Dench was born in York, England, on December 9, 1934. York is a historic city in the county of North Yorkshire. The answer is \\boxed{York, England}"
        }
        cot_ex['example2'] = {
            "Question": "From which country did Angola achieve independence in 1975?",
            "Answer": "To determine from which country Angola achieved independence in 1975, let’s analyze step by step:\n1. Understand Angola's colonial history: Angola was a colony for several centuries before gaining independence. Understanding its colonizing power will clarify this.\n2. Identify the European colonizing country: Angola was colonized by Portugal during the period of European exploration and colonization in Africa. Portuguese influence in Angola began in the late 15th century and solidified over the subsequent centuries.\n3. Independence movements in Africa: Many African nations gained independence during the mid-20th century. Angola's independence was part of a broader decolonization wave.\n4. Key events leading to independence: Angola’s path to independence included a prolonged struggle marked by an anti-colonial war (1961–1974). The Carnation Revolution in Portugal in 1974 significantly weakened Portuguese control over its colonies.\n5. Outcome: On November 11, 1975, Angola officially declared its independence from Portugal.\nThus, Angola achieved independence from Portugal in 1975. Therefore, the answer is \\boxed{Portugal}"
        }    
        
    elif dataset == "truthfulqa":
        if emphasize:
            cot_ex['example1'] = {
                "Question": "What is human life expectancy in the United States?",
                "Answer": (
                    "To determine the life expectancy in the United States, we consider multiple factors "
                    "including healthcare quality, demographic trends, and statistical studies. Several "
                    "reliable reports consistently indicate an average of about 78 years. "
                    "**In conclusion, human life expectancy in the United States is 78 years.**"
                )
            }

            cot_ex['example2'] = {
                "Question": "Who was president of the United States in 1955?",
                "Answer": (
                    "Historical records and trusted sources on U.S. presidential history show that during 1955, "
                    "the presidency was held by Dwight D. Eisenhower. He led the nation through various key events "
                    "of the time. "
                    "**In conclusion, Dwight D. Eisenhower was the president in 1955.**"
                )
            }
    return cot_ex