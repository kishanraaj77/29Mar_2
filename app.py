from aiogram import Bot, Dispatcher, types
from transformers import pipeline
from aiogram.utils import executor
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
#from sklearn.externals import joblib
import json 
import requests
import time
import urllib
import joblib
#import cdqa
 

bot = Bot(token='1786706698:AAEIoqj0tgJYHCYiAx5QIYm-F3wxR18-Ys0')
dp = Dispatcher(bot)

@dp.message_handler(commands= ['start','help'])
async def main(message:types.Message):
    await message.reply("this is from aiogram bot , Welcome")

@dp.message_handler(commands= ['hi'])
async def main(message:types.Message):
    await message.reply("this is HI from aiogram bot , Welcome HI")

@dp.message_handler(commands= ['qna'])
async def main(message:types.Message):
    question_answering = pipeline("question-answering")
    context = """Machine learning (ML) is the study of computer algorithms that improve automatically
    through experience. It is seen as a part of artificial intelligence. 
    Machine learning algorithms build a model based on sample data, known as "training data", 
    in order to make predictions or decisions without being explicitly programmed to do so. 
    Machine learning algorithms are used in a wide variety of applications, 
    such as email filtering and computer vision, 
    where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks."""
    question = "What are machine learning models based on?"
    result = question_answering(question=question, context=context)
    await message.reply(text=result)

@dp.message_handler(commands= ['engtext'])
async def main(message:types.Message):
    from transformers import pipeline
    txt_gner = pipeline("text-generation")
    text_input = "mount everest is "
    txt_output = txt_gner(text_input, max_length = 50, do_sample=False)[0]
    await message.reply(text= txt_output)

@dp.message_handler(commands= ['chintext'])
async def main(message:types.Message):
    from transformers import pipeline , BertTokenizerFast, AutoModelWithLMHead
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModelWithLMHead.from_pretrained('ckiplab/gpt2-base-chinese')
    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer)
    text_input  = "机器学习是"
    generated_text = text_generation(text_input, max_length = 50, do_sample=False)[0]
    await message.reply(text= generated_text)

@dp.message_handler(commands= ['qnabert'])
async def main(message:types.Message):
    import torch
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    Question = 'The purpose of the NewsQA dataset'
    paragrah = 'With massive volumes of written text being produced every second, how do we make sure that we have the most recent and relevant information available to us? Microsoft research Montreal is tackling this problem by building AI systems that can read and comprehend large volumes of complex text in real-time. The purpose of the NewsQA dataset is to help the research community build algorithms that are capable of answering questions requiring human-level comprehension and reasoning skills.'
    encoding = tokenizer.encode_plus(text = Question, text_pair=paragrah, add_special = True)
    # token embedding 
    inputs = encoding['input_ids']
    #3 segment embedgin 
    sentence_embed = encoding['token_type_ids'] 
    # input tokens 
    tokens = tokenizer.convert_ids_to_tokens(inputs) 
    start_scores, end_scores  = model(input_ids=torch.tensor([inputs]), token_type_ids = torch.tensor([sentence_embed]), return_dict=False)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])

    await message.reply(text=answer)

async def setup_bot_commands(dispatcher: Dispatcher):
    """
    Here we setup bot commands to make them visible in Telegram UI
    """
    bot_commands = [
        types.BotCommand(command="/qna", description="this is for QnA "),
        types.BotCommand(command="/help", description="Help and source code"),
        types.BotCommand(command="/engtext", description="This is for English Text genaration"),
        types.BotCommand(command="/chintext", description="This is for Chinese Text genaration"),
        types.BotCommand(command="/qnabert", description="This is for QnA Bert for news Dataset"),
        

    ]
    await bot.set_my_commands(bot_commands)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True, on_startup=setup_bot_commands)


