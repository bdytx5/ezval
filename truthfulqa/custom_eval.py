# import pandas as pd
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from evaluate import load

# def evaluate_gpt2_on_truthfulqa(data_path, model_name="gpt2"):
#     # Load the TruthfulQA dataset
#     data = pd.read_csv(data_path)

#     # Initialize the GPT-2 model and tokenizer
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#     # Move the model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the evaluation metrics
#     bleurt = load("bleurt")
#     rouge = load("rouge")
#     bleu = load("bleu")

#     # Evaluate the model on each question-answer pair
#     results = []
#     for _, row in data.iterrows():
#         question = row["Question"]
#         true_answers = eval(row["Best Answer"])  # Assuming the true answers are stored as a list in the CSV
#         false_answers = eval(row["Incorrect Answer"])  # Assuming the false answers are stored as a list in the CSV

#         # Tokenize the question and answers
#         question_tokens = tokenizer.encode(question, return_tensors="pt").to(device)
#         true_answer_tokens = [tokenizer.encode(ans, return_tensors="pt").to(device) for ans in true_answers]
#         false_answer_tokens = [tokenizer.encode(ans, return_tensors="pt").to(device) for ans in false_answers]

#         # Calculate log probabilities for true and false answers
#         true_scores = []
#         false_scores = []

#         with torch.no_grad():
#             for ans_tokens in true_answer_tokens:
#                 outputs = model(question_tokens, labels=ans_tokens)
#                 log_probs = outputs.logits.log_softmax(dim=-1)
#                 ans_log_probs = log_probs[0, -ans_tokens.shape[1]:].diagonal()
#                 true_scores.append(ans_log_probs.sum().item())

#             for ans_tokens in false_answer_tokens:
#                 outputs = model(question_tokens, labels=ans_tokens)
#                 log_probs = outputs.logits.log_softmax(dim=-1)
#                 ans_log_probs = log_probs[0, -ans_tokens.shape[1]:].diagonal()
#                 false_scores.append(ans_log_probs.sum().item())

#         # Calculate evaluation metrics
#         lprob_scores_true = true_scores
#         lprob_scores_false = false_scores
#         lprob_max = max(true_scores)
#         lprob_diff = max(true_scores) - max(false_scores)

#         # Generate the model's answer
#         model_answer = model.generate(question_tokens, max_length=100, num_return_sequences=1, early_stopping=True)
#         model_answer = tokenizer.decode(model_answer[0], skip_special_tokens=True)

#         # Calculate BLEURT, ROUGE, and BLEU scores
#         bleurt_scores = [bleurt.compute(predictions=[model_answer], references=[ans]) for ans in true_answers]
#         rouge_scores = [rouge.compute(predictions=[model_answer], references=[ans]) for ans in true_answers]
#         bleu_scores = [bleu.compute(predictions=[model_answer], references=[[ans]]) for ans in true_answers]

#         result = {
#             "Question": question,
#             "True Answers": true_answers,
#             "False Answers": false_answers,
#             "Model Answer": model_answer,
#             "lprob scores-true": lprob_scores_true,
#             "lprob scores-false": lprob_scores_false,
#             "lprob max": lprob_max,
#             "lprob diff": lprob_diff,
#             "BLEURT Scores": bleurt_scores,
#             "ROUGE Scores": rouge_scores,
#             "BLEU Scores": bleu_scores
#         }
#         results.append(result)

#     # Create a DataFrame with the evaluation results
#     results_df = pd.DataFrame(results)
#     return results_df

# # Specify the path to your TruthfulQA dataset
# data_path = "/Users/brettyoung/Desktop/dev_24/dpo/evals/TruthfulQA/TruthfulQA.csv"

# # Evaluate GPT-2 on TruthfulQA
# evaluation_results = evaluate_gpt2_on_truthfulqa(data_path)

# # Print the evaluation results
# print(evaluation_results)



import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from evaluate import load


def evaluate_gpt2_on_truthfulqa(data_path, model_name="gpt2"):
    # Load the TruthfulQA dataset
    data = pd.read_csv(data_path)

    # Initialize the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the evaluation metrics
    bleurt = load("bleurt")
    rouge = load("rouge")
    bleu = load("bleu")

    # Evaluate the model on each question-answer pair
    results = []
    for _, row in data.iterrows():
        question = row["Question"]
        best_answer = row["Best Answer"]  # Assuming the best answer is stored as a string in the CSV
        correct_answers = [ans.strip() for ans in row["Correct Answers"].split(";")]  # Split correct answers by semicolon and strip whitespace
        incorrect_answers = [ans.strip() for ans in row["Incorrect Answers"].split(";")]  # Split incorrect answers by semicolon and strip whitespace

        # Tokenize the question and answers
        question_tokens = tokenizer.encode(question, return_tensors="pt").to(device)
        best_answer_tokens = tokenizer.encode(best_answer, return_tensors="pt").to(device)
        correct_answer_tokens = [tokenizer.encode(ans, return_tensors="pt").to(device) for ans in correct_answers]
        incorrect_answer_tokens = [tokenizer.encode(ans, return_tensors="pt").to(device) for ans in incorrect_answers]

        # Calculate log probabilities for best, correct, and incorrect answers
        best_score = model(question_tokens, labels=best_answer_tokens).logits.log_softmax(dim=-1).diagonal().sum().item()
        correct_scores = []
        incorrect_scores = []

        with torch.no_grad():
            for ans_tokens in correct_answer_tokens:
                outputs = model(question_tokens, labels=ans_tokens)
                log_probs = outputs.logits.log_softmax(dim=-1)
                ans_log_probs = log_probs[0, -ans_tokens.shape[1]:].diagonal()
                correct_scores.append(ans_log_probs.sum().item())

            for ans_tokens in incorrect_answer_tokens:
                outputs = model(question_tokens, labels=ans_tokens)
                log_probs = outputs.logits.log_softmax(dim=-1)
                ans_log_probs = log_probs[0, -ans_tokens.shape[1]:].diagonal()
                incorrect_scores.append(ans_log_probs.sum().item())

        # Calculate evaluation metrics
        lprob_scores_best = [best_score]
        lprob_scores_correct = correct_scores
        lprob_scores_incorrect = incorrect_scores
        lprob_max = max(lprob_scores_correct)
        lprob_diff = lprob_max - max(lprob_scores_incorrect)

        # Generate the model's answer
        model_answer = model.generate(question_tokens, max_length=100, num_return_sequences=1, early_stopping=True)
        model_answer = tokenizer.decode(model_answer[0], skip_special_tokens=True)

        # Calculate BLEURT, ROUGE, and BLEU scores
        bleurt_score = bleurt.compute(predictions=[model_answer], references=[best_answer])
        rouge_score = rouge.compute(predictions=[model_answer], references=[best_answer])
        bleu_score = bleu.compute(predictions=[model_answer], references=[[best_answer]])

        result = {
            "Question": question,
            "Best Answer": best_answer,
            "Correct Answers": correct_answers,
            "Incorrect Answers": incorrect_answers,
            "Model Answer": model_answer,
            "lprob scores-best": lprob_scores_best,
            "lprob scores-correct": lprob_scores_correct,
            "lprob scores-incorrect": lprob_scores_incorrect,
            "lprob max": lprob_max,
            "lprob diff": lprob_diff,
            "BLEURT Score": bleurt_score,
            "ROUGE Score": rouge_score,
            "BLEU Score": bleu_score
        }
        results.append(result)

    # Create a DataFrame with the evaluation results
    results_df = pd.DataFrame(results)
    return results_df

# Specify the path to your TruthfulQA dataset
data_path = "/Users/brettyoung/Desktop/dev_24/dpo/evals/TruthfulQA/TruthfulQA.csv"

# Evaluate GPT-2 on TruthfulQA
evaluation_results = evaluate_gpt2_on_truthfulqa(data_path)

# Print the evaluation results
print(evaluation_results)
