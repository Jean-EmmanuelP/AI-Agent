import os
import re
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement depuis .env
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Hello world",
        }
    ],
    model="llama-3.3-70b-versatile",
)

# print(chat_completion.choices[0].message.content)
class Agent: 
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=self.messages)
        return completion.choices[0].message.content
prompt= """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked
Use Action to run one of the actions available to you – then return PAUSE
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number – uses Python so be sure to use valid syntax

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
def calculate(what):
    return eval(what)
def average_dog_weight(name):
    if (name) in "Scottish Terrier":
        return("Scottish Terrier average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
abot = Agent(prompt)
result = abot("How much does a toy poodle weigh?")
result = average_dog_weight("Toy Poodle")
# print(result)
next_prompt = "Observation: {}".format(result)
abot(next_prompt)
# print(next_prompt)
# print('-------')
# print(abot(next_prompt))
# print('-------')
# print(abot.messages)
question = """I have 2 dogs, a border collie and a scottish terrier. What is their combined weight"""
result = abot(question)
# print(result)
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
# print(abot(next_prompt))
# print(abot(next_prompt))
next_prompt = average_dog_weight("Border Collie")
# print(abot(next_prompt))
next_prompt = "Observation: {}".format(eval("37 + 20"))
# print(next_prompt)
# print(abot(next_prompt))
action_re = re.compile(r'^Action: (\w+): (.*)$')
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a.strip())
            for a in result.split('\n')
            if action_re.match(a.strip())
        ]
        if actions:
            # There is an action to run
            action_match = actions[0]
            action = action_match.group(1)
            action_input = action_match.group(2)
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {}: {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation: ", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)