"""
This has all the APIs for calculating Log-Likelihood for a given sentence using an LLM

"""
import torch
import math
from nltk import tokenize
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LLMInference:

    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

    def gpt3_inference(self, prompt, index):
        """

        :param prompt: Input prompt to be fed to GPT-3
        :param index: starting of the index from which log likelihood has to be calculated
        :return: summation of log likelihood of target sentence
        """
        openai.api_key = ""  # Add the key for GPT-3

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=1,
            logprobs=1,
            echo=True
        )
        logprobs = response.choices[0].logprobs.token_logprobs
        return sum(logprobs[index: len(logprobs) - 1])

    def gpt2_log_likelihood(self, sentence, topic_to_be_conditioned_on, previous_sentences_to_be_conditioned_on,
                            model="gpt2"):
        """

        :param sentence: sentence for which the log likelihood has to be calculated
        :param topic_to_be_conditioned_on: conditional prompt topic
        :param previous_sentences_to_be_conditioned_on: contextual prompt
        :param model: model to be used (GPT-2 or GPT-3)
        :return: Log Likelihood of the sentence
        """
        conditional_prompt = topic_to_be_conditioned_on
        if previous_sentences_to_be_conditioned_on and len(previous_sentences_to_be_conditioned_on) > 0:
            conditional_prompt += " " + " ".join(previous_sentences_to_be_conditioned_on)

        target_sentence_log_likelihood = 0
        conditional_prompt_tokens = self.gpt2_tokenizer.encode(conditional_prompt, return_tensors="pt")
        prompt_tokens_length = len(conditional_prompt_tokens[0])
        target_sentence_tokens = self.gpt2_tokenizer.encode(sentence, return_tensors="pt")
        prompt_and_target_combined_token = torch.cat((conditional_prompt_tokens, target_sentence_tokens), 1)

        if model == "gpt3":
            return self.gpt3_inference(conditional_prompt + " " + sentence, prompt_tokens_length)

        output = self.gpt2_model(prompt_and_target_combined_token)
        logits = output[0]
        with torch.no_grad():

            for i in range(prompt_tokens_length, len(prompt_and_target_combined_token[0])):
                next_token_prob = torch.softmax(logits[:, i - 1, :], dim=-1)
                next_token_index = prompt_and_target_combined_token[0][i]
                token_likelihood = next_token_prob[0][next_token_index].item()
                target_sentence_log_likelihood += math.log(token_likelihood, 2)

        return target_sentence_log_likelihood

    def compute_story_metric(self, sentences, topic, history_size=math.inf, model="gpt2"):
        """

        :param sentences: Entire story as a string
        :param topic: Summary of the given story
        :param history_size: History size for contextual length
        :param model: model to be used
        :return:
        """
        if not model:
            model = "gpt2"
        if not history_size:
            history_size = math.inf
        sentence_in_story = tokenize.sent_tokenize(sentences)
        topic_driven_output = []
        contextual_output = []

        for i in range(len(sentence_in_story)):
            print(f"Processing sentence {i + 1} out of {len(sentence_in_story)}")
            topic_driven_output.append(self.gpt2_log_likelihood(sentence_in_story[i], topic, None, model))
            history_end = i
            history_start = max(0, i - history_size)
            contextual_output.append(
                self.gpt2_log_likelihood(sentence_in_story[i], topic, sentence_in_story[history_start:history_end],
                                         model))
        sequentiality = 0

        for i in range(len(sentence_in_story)):
            sequentiality += (topic_driven_output[i] - contextual_output[i]) / (-1 * len(sentence_in_story[i]))

        return (sequentiality / len(sentence_in_story)), topic_driven_output, contextual_output


if __name__ == "__main__":
    topic = "I wrote about a visit from my mother-in-law, who has advanced stage cancer. I discuss seeing her and how I perceived her when I saw her, as well as how this diagnosis has affected me personally and made me reflect on her death and mortality in general."
    imagined_story = "For my 48th birthday, which was 4 months ago, my friend Lauren invited me to a wine bar. It's near the old cathedral and serves organic wine. It had just opened and was on a charming street with old style street lamps and cobble stones and had a comfortable, cozy and dark atmosphere inside. I had wanted to try it and it was good to see her. I didn't realize that she had invited several other friends of hers, people I know a bit but not too well. I wasn't planning on drinking as much as I did, thinking that she and I could have a couple of glasses of wine each and go home. But the conversation took a life of its own, the bottles of wine were ordered and drunk. I was enjoying myself and did not realize how much wine I had to drink. Lauren ordered a charcuterie board as well, which was delicious and seemed to encourage more sips of wine (to go with the cheese, of course). We were laughing and gossiping about the other colleagues in our department. Before I knew it I was drunk and feeling unstable on my legs. We all were and only left once the bar closed. I have no idea how I got home. Cabs are expensive so I must have braved the train late at night on my own, which is not safe. The next morning I woke up and felt terrible. My mouth was like cotton wool, I was dehydrated and my skin looked blotchy. That's it, I concluded! This cannot happen again. I need to detox, step back from drinking so much and take better care of myself. No more late nights alone on the train without the use of my full faculties. It has been surprisingly easy. I thought I would crave wine but I do not miss it or the hangovers. I appreciate feeling good, hydrated, and have been running and exercising more consistently. I wish I had done it sooner and it will be easier to limit the wine going forward, I really believe."
    retold_story = "My mother-in-law was relocating to a new state to undergo treatment for advanced stage cancer. She stopped to stay with us during her journey. In the past I had not always felt close to her. She is a very exacting, highly intelligent and headstrong woman. I admire her for that but it can be challenging as she can be curt. Over time I have felt a shift and in this visit, I felt that shift very strongly. She was softer, kinder and more vulnerable. She looked frail, thin. She was more patient and loving and was accepting of my recent unorthodox career choices. I felt myself opening up to her more and feeling at ease in saying what I wanted to. I saw that she received it as it was, without judgment. When we said goodbye to her early in the morning, she looked at both my husband and I very pointedly. I felt that this moment was significant. I felt that we all understood that her life hung in the balance. We all knew that we did not know or have any control, beyond the limits of modern medicine. I felt that I understood something about mortality through her, that we all are hanging in the balance and vulnerable at any given moment to death. As she waved goodbye I felt an overwhelming love for her and both an enormous appreciation for life while understanding my incredible impotence in the face of death."
    recalled_story = "Since my own mother died when I was younger, I have become close to my husband's mother and maintain an affectionate correspondence with her. She is not my mom but I love her for who she is all the same. In the past few months my mother-in-law has been quite ill. In the last month, I learned that she has stage 2 borderline stage 3 lung cancer involving a large mass in the lung. I saw her last week for the first time since her diagnosis when she stopped to spend a night with us while on her to begin cancer treatment. She was in good spirits despite the long journey. She joked about death and mentioned doing her 'dying person' prayers. Being a feminist and an incredibly open-minded person, she was extremely supportive towards me and my unconventional career choices, encouraging me. Everything felt suspended in disbelief and she was filled with more tenderness than usual. In a way this surprised me because she has many reasons to be angry: with her doctors or with life in general for striking her with cancer, in addition to many other misfortunes. She was relaxed and calm and put on a good face but she also looked frail and had lost a significant amount of weight. The next morning she got up early and when we said goodbye to her, she seemed so small and impotent and alone. She looked at both of us directly - her son and me - and somehow I felt that we all understood what is at stake here. As she drove off I wondered if I would see her again. Her visit and seeing her has made me feel many things. I grieve that she has to face this terrible disease and I feel a deep sorrow for my husband, who has been sleeping poorly, worried and more quiet than usual. He would not be who he is today -- respectful towards women, kind, loving, patient -- if it were not for his mother. He does not want to lose her. I feel a sense of total powerlessness in the face of death and the passage of time. I think about how my mother-in-law has lived this amazing, full life, always doing what she wanted even when it was the controversial choice. I think about the vast emptiness of not being able to talk to her again or to be able to email her or see her. I think about how I could have visited her last year but chose not to because it was too cold and rainy where she lived, and I assumed I would have many more chances to see her. I think about how all of us are suspended in disbelief in every moment, unsure about when death will happen to us. It is incredibly overwhelming and I try to be as kind as I can to myself and to everyone around me, it is all I can do, it seems. I do not know what else I can do."

    model = LLMInference()

    print("Imagined", model.compute_story_metric(imagined_story, topic, model="gpt2"))
    print("Recalled", model.compute_story_metric(recalled_story, topic, model="gpt2"))
    print("Retold", model.compute_story_metric(retold_story, topic, model="gpt2"))
