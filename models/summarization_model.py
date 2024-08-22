# models/summarization_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM

class SummarizationModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.summarizer = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    def summarize(self, text):
        prompt = f"Summarize the following process in approximately 150 words: {text}"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.summarizer.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not summary.endswith('.'):
                summary += '.'

            return summary

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary could not be generated due to an error."
