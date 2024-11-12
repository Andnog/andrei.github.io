---
title: Introduction to Azure OpenAI
summary: "Tutorial: Explaining ChatGPT with Azure OpenAI Integration"
date: 2024-02-19
type: docs
math: false
weight: 1
tags:
  - LLM
  - Azure Open AI
  - ChatGPT
image:
  caption: 'Azure Open AI'
---

## General Description

Welcome to the **ChatGPT API Tutorial**—an immersive guide for integrating **OpenAI's ChatGPT into your applications using the API with Azure OpenAI**. Whether you're a developer aiming to enhance user interactions or a data enthusiast exploring language model capabilities, this tutorial provides the insights you need to leverage ChatGPT effectively. 

[📘 Complete notebook tutorial](https://github.com/Andnog/LLM-Tutorials/blob/main/1.0%20ChatGPT%20API.ipynb)

---

## 🤔 What is ChatGPT?

### 🧠 What is ChatGPT?

ChatGPT is a powerful language model developed by OpenAI, capable of generating human-like text based on given prompts. It excels in natural language understanding and can be harnessed for various applications, from content creation to conversational interfaces.

ChatGPT is built upon the GPT (Generative Pre-trained Transformer) architecture, a state-of-the-art language model developed by OpenAI. GPT models are based on transformer neural networks, utilizing attention mechanisms for contextual understanding. In the case of ChatGPT, the model is specifically fine-tuned for conversational contexts.

### 🛠️ How does ChatGPT work?

Behind ChatGPT's capabilities is a vast neural network that has undergone extensive pre-training on diverse internet text. This pre-training equips the model with an inherent understanding of grammar, context, and a wide range of topics. The transformer architecture, with attention mechanisms, enables ChatGPT to grasp contextual relationships, crucial for generating coherent and contextually relevant responses.

The ChatGPT API serves as the gateway to unlocking the model's potential programmatically. Through HTTP requests, users can engage with ChatGPT by providing a series of messages, each with a designated role ('system', 'user', or 'assistant') and content. The API facilitates dynamic conversations, allowing users to instruct the 
model and receive tailored responses.

### 🌐 Use Cases and Applications

The versatility of ChatGPT makes it applicable across various domains. Explore use cases such as content creation, code generation, conversational agents, and more. Whether you're looking to enhance user interactions or automate certain tasks, ChatGPT's applications are diverse and impactful.

___
## 🛠️ How to Use OpenAI

### 🚀 Installation Guide

#### ⚙️ Prerequisites

Before diving into using ChatGPT via the OpenAI package, ensure you have the following prerequisites installed:

- Python (version 3.6 or higher)
- OpenAI Python library (`openai`)

You can install the OpenAI library using:

```bash
pip install openai
```

### 📝 Practical Examples

#### 👋 Basic Usage

Get started with a simple example to make your first interaction with ChatGPT:

```python
import openai

openai.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]
)
```

This example demonstrates a basic interaction, where the system message sets the assistant's behavior, and the user message instructs the model.

___
#### 🔗 Advanced Usage with Azure OpenAI

For users working with Azure, you can use the `AzureOpenAI` client. Here is an example:

```python
from azure_openai import AzureOpenAI
import os

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_KEY"),  
  api_version = os.getenv("api_version"),
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]
)

print("ChatGPT Response:")
print(response.choices[0].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 ChatGPT Response: <br>
 The Los Angeles Dodgers won the World Series in 2020.
</blockquote>


___
## 🔧 Hyperparameters

Hyperparameters play a crucial role in shaping the behavior of ChatGPT. Let's explore key hyperparameters and their impact on model responses:

| Parameter         | Description                                             | Range            |
|--------------------|---------------------------------------------------------|------------------|
| temperature        | Controls the level of randomness in the generated responses. A higher value (closer to 1.0) increases creativity but may introduce randomness. | [0.0, 1.0]       |
| max_tokens         | Sets the maximum limit of tokens in the generated response, limiting its length.  | [1, ∞]           |
| `top_p`              | Controls the diversity of responses. A higher value (closer to 1.0) allows for a broader range of words in the output. | [0.0, 1.0]       |
| presence_penalty   | Controls the preference for or against including certain words in the response. A higher value (closer to 1.0) encourages more mention of specified words. | [0.0, 1.0]       |
| frequency_penalty  | Controls the preference for or against repeating words in the response. A higher value (closer to 1.0) discourages repeated words. | [0.0, 1.0]       |
| stop_sequence      | Specifies a sequence that, when encountered, stops the generation of the response. | Any sequence     |
| best_of            | Specifies the number of candidate responses to consider before selecting the best one. | [1, ∞]           |
| n                  | Controls the number of responses generated for a single prompt. | [1, ∞]           |
| log_level          | Controls the logging level of output, allowing adjustment between detailed debugging information and critical error messages. | "debug", "info", "warning", "error", "critical" |

**Detail explanation**

- **Temperature**: In short, the lower the `temperature`, the more deterministic the results in the sense that the highest probable next token is always picked. Increasing temperature could lead to more randomness, which encourages more diverse or creative outputs. You are essentially increasing the weights of the other possible tokens. In terms of application, you might want to use a lower temperature value for tasks like fact-based QA to encourage more factual and concise responses. For poem generation or other creative tasks, it might be beneficial to increase the temperature value.

- **Top P**: A sampling technique with `temperature`, called nucleus sampling, where you can control how deterministic the model is. If you are looking for exact and factual answers keep this low. If you are looking for more diverse responses, increase to a higher value. If you use Top P it means that only the tokens comprising the `top_p` probability mass are considered for responses, so a low `top_p` value selects the most confident responses. This means that a high `top_p` value will enable the model to look at more possible words, including less likely ones, leading to more diverse outputs. The general recommendation is to alter temperature or Top P but not both.

- **Max Length**: You can manage the number of tokens the model generates by adjusting the `max length`. Specifying a max length helps you prevent long or irrelevant responses and control costs.

- **Stop Sequences**: A `stop sequence` is a string that stops the model from generating tokens. Specifying stop sequences is another way to control the length and structure of the model's response. For example, you can tell the model to generate lists that have no more than 10 items by adding `"11"` as a stop sequence.

- **Frequency Penalty**: The `frequency penalty` applies a penalty on the next token proportional to how many times that token already appeared in the response and prompt. The higher the frequency penalty, the less likely a word will appear again. This setting reduces the repetition of words in the model's response by giving tokens that appear more a higher penalty.

- **Presence Penalty**: The `presence penalty` also applies a penalty on repeated tokens but, unlike the frequency penalty, the penalty is the same for all repeated tokens. A token that appears twice and a token that appears 10 times are penalized the same. This setting prevents the model from repeating phrases too often in its response. If you want the model to generate diverse or creative text, you might want to use a higher presence penalty. Or, if you need the model to stay focused, try using a lower presence penalty.

Similar to `temperature` and `top_p`, the general recommendation is to alter the frequency or presence penalty but not both.

*{{< icon name="python" >}} Example:*
```python
prompt_text = """
You are a helpful assistant.
"""

response = client.chat.completions.create(
    model=os.getenv("model_name"),
    messages=[
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature=0.2,
    max_tokens=15,
)

print("Output:")
print(response.choices[0].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 Output: <br>
 Why did the tomato turn red? Because it saw the salad dressing!
</blockquote>

*{{< icon name="python" >}} Example:*
```python
prompt_text = """
You are a helpful assistant.
"""

response = client.chat.completions.create(
    model=os.getenv("model_name"),
    messages=[
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature = 0.5,
    max_tokens = 50,
    top_p = 0.8,
    presence_penalty = 0.5,
    frequency_penalty = 0.3
)

print("Output:")
print(response.choices[0].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 Output: <br>
 Why don't scientists trust atoms? <br>
 Because they make up everything.
</blockquote>

*{{< icon name="python" >}} Example:*
```python
prompt_text = """
You are a helpful assistant.
"""

response = client.chat.completions.create(
    model=os.getenv("model_name"),
    messages=[
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature = 1.2,
    n = 2,
    top_p = 0.9,
)

print("Output:")
print(response.choices[0].message.content)
print(response.choices[1].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 Output: <br>
 Why did the tomato turn red? Because it saw the salad dressing! <br>
 Why did the tomato turn red? Because it saw the salad dressing!
</blockquote>


___
## 📏 Parameters and Structure

#### 📃 Guiding Principles

- **Specificity:** Be explicit and detailed in your prompts. Clearly communicate the desired format or structure for the response.

- **Contextualization:** Provide context in user messages to guide the model's understanding. System messages set the tone for the assistant's behavior.

- **Iterative Conversation:** Build conversations with multiple turns. This helps the model maintain context and produce coherent responses.

#### 💬 Model Selection

- **Model:** Specify the ChatGPT model to use. For example, "gpt-3.5-turbo."

#### 📚 Message Structure

- **Messages:** Compose an array of messages, each with a role ("system," "user," or "assistant") and content (text of the message). Order matters, with the conversation evolving sequentially.

#### 🔧 System Message

- **Role:** "system"
- **Content:** Set the assistant's behavior with a concise system message at the beginning of the conversation.

#### 🙍‍♂️ User Messages

- **Role:** "user"
- **Content:** Clearly instruct the model in user messages. Use detailed language and iterate the conversation for context.

#### 🩺 Assistant Messages

- **Role:** "assistant"
- **Content:** Use assistant messages to provide additional context or information to guide the model's understanding.

___
### ⚠️ Troubleshooting and Common Challenges

**Unintended Output**

If the model provides unintended or undesirable responses:

- Refine prompts: Experiment with different phrasing and provide more explicit instructions.
  
- Adjust hyperparameters: Fine-tune temperature, max tokens, and other settings for desired behavior.

**Response Length**

For controlling response length:

- Utilize max tokens: Set an appropriate max tokens value to limit the length of responses.

*{{< icon name="python" >}} Example:*
```python
response = client.chat.completions.create(
    model=os.getenv("model_name"),
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."}
        ,
        {
            "role": "user", 
            "content": "Who won the world series in 2020?"}
        ,
        {
            "role": "assistant", 
            "content": "The Los Angeles Dodgers won the World Series in 2020."}
        ,
        {
            "role": "user", 
            "content": "Tell me more about their journey to victory."}
        ,
        {
            "role": "assistant", 
            "content": "The Dodgers had a remarkable season, overcoming challenges and demonstrating exceptional teamwork."}
        ,
        {
            "role": "user", 
            "content": "What were the key highlights of the final game?"
        }
    ]
)

print("ChatGPT Response:")
print(response.choices[0].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 ChatGPT Response: <br>
 In the final game, the Dodgers defeated the Tampa Bay Rays with a score of 3-1 to take their first World Series title since 1988. The game was tightly contested, with both teams playing great defense and pitching. The Dodgers took an early lead in the 1st inning with a solo home run by Mookie Betts, who was a key player throughout the playoffs. The Rays tied the game in the 4th inning, but the Dodgers quickly retook the lead with a sacrifice fly by Corey Seager in the 6th inning. Pitcher Julio Urías then came in to record the final seven outs and secure the championship for the Dodgers.
</blockquote>

___
## 🧠 Models

### 🧪 Latest Models

OpenAI offers several models, each catering to specific needs. Let's explore the latest models as of the provided information:

**GPT-4 and GPT-4 Turbo**

- **GPT-4:** A large multimodal model with improved reasoning capabilities, broad general knowledge, and the ability to accept text or image inputs. Suitable for complex problem-solving and accurate responses.

- **GPT-4 Turbo:** An optimized version of GPT-4 for chat applications, providing high accuracy and cost-effectiveness. Well-suited for traditional completions tasks using the Chat Completions API.

**GPT-4 Turbo Variants**

- **gpt-4-1106-preview:** The latest GPT-4 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. Note: This preview model is not yet suited for production traffic.

- **gpt-4-vision-preview:** GPT-4 Turbo with vision capabilities, enabling the understanding of images in addition to text inputs. A preview model version not yet suited for production traffic.

**GPT-3.5**

GPT-3.5 models, including the optimized version gpt-3.5-turbo, offer a balance of natural language understanding and generation.

**GPT-3.5 Turbo Variants**

- **gpt-3.5-turbo-1106:** The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens.

- **gpt-3.5-turbo:** The optimized version for chat applications, offering a cost-effective and capable solution. Well-suited for both chat and traditional completions tasks.

- **gpt-3.5-turbo-16k:** Similar to gpt-3.5-turbo but with a larger context window of 16,385 tokens.

- **gpt-3.5-turbo-instruct:** Designed for compatibility with legacy Completions endpoint and not Chat Completions.

**Choosing the Right Model**

- **Task Complexity:** For general tasks, gpt-3.5-turbo is often sufficient. GPT-4 variants excel in more complex reasoning scenarios.

- **Multilingual Capabilities:** GPT-4 outperforms previous models and demonstrates strong performance in multiple languages.

**Integration Example**

*{{< icon name="python" >}} Example:*
```python
from azure_openai import AzureOpenAI
import os

client = AzureOpenAI(
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version=os.getenv("api_version"),
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model="gpt-4-1106-preview",  # Specify the desired model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)

print("ChatGPT Response:")
print(response.choices[0].message.content)

# Note: For using other model versions it is necessary to upload the settings for the corresponding model: Endpoint and APIKey
```

___
## 🔍 Good Practices

**Optimizing Interactions with ChatGPT**

As you engage with ChatGPT, adopting good practices can enhance the quality and effectiveness of your interactions. Let's explore key practices to optimize your experience:

**1. Contextualize Conversations**

- **Why:** Provide context to guide the model's understanding.
- **How:** Use system and user messages to build a coherent conversation. Reference prior messages for context continuity.

**2. Experiment with Prompt Iterations**

- **Why:** Refine prompts for better results.
- **How:** Experiment with different phrasing, structures, and levels of detail. Iterate to find the most effective prompt.

**3. Control Response Length**

- **Why:** Manage output length for concise and relevant responses.
- **How:** Utilize the `max_tokens` parameter to set a limit on the length of generated responses.

**4. Fine-Tune Hyperparameters**

- **Why:** Adjust model behavior to meet specific requirements.
- **How:** Experiment with hyperparameters like temperature, presence_penalty, and frequency_penalty to achieve desired response characteristics.

**5. Utilize System Messages Effectively**

- **Why:** Set the behavior of the assistant for improved guidance.
- **How:** Craft concise and clear system messages at the beginning of the conversation to influence the assistant's behavior.

**6. Handle Unintended Output**

- **Why:** Address unexpected or undesirable responses.
- **How:** Refine prompts, adjust hyperparameters, and experiment with system/user messages to improve output.


*{{< icon name="python" >}} Example:*
```python
from azure_openai import AzureOpenAI
import os

client = AzureOpenAI(
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version=os.getenv("api_version"),
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user", 
            "content": "Tell me about the impact of climate change on polar bears."
        },
        {
            "role": "assistant", 
            "content": "Climate change poses significant threats to polar bears, affecting their habitat and food sources."
        },
        {
            "role": "user", 
            "content": "How can we mitigate these impacts?"
        },
    ]
)

print("ChatGPT Response:")
print(response.choices[0].message.content)
```

In this example, contextualizing the conversation, experimenting with prompt iterations, and utilizing effective system messages contribute to a more informative and relevant response.

By incorporating these good practices, you can optimize your interactions with ChatGPT and obtain more tailored and valuable outputs.

*{{< icon name="python" >}} Example:*
```python
response = client.chat.completions.create(
    model=os.getenv("model_name"),
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user", 
            "content": "Tell me about the impact of climate change on polar bears."
        },
        {
            "role": "assistant", 
            "content": "Climate change poses significant threats to polar bears, affecting their habitat and food sources."
        },
        {
            "role": "user", 
            "content": "How can we mitigate these impacts?"
        },
    ]
)

print("ChatGPT Response:")
print(response.choices[0].message.content)
```

<blockquote>
 {{< icon name="code-bracket" >}} Output: <hr class="m-0-imp"> 
 ChatGPT Response: <br>
 Mitigating the impacts of climate change on polar bears and their habitat will require significant global efforts to reduce greenhouse gas emissions and slow the rate of warming. Some other steps that can help include: <br>

1. Reducing carbon emissions: This can be achieved through a range of measures, including using more renewable energy sources, improving energy efficiency, reducing reliance on fossil fuels, and adopting sustainable transportation. <br>

2. Preserving habitats: Polar bears rely on sea ice for hunting, resting, and traveling, protecting important habitat areas can help preserve their health and safety. <br>

3. Promoting sustainable tourism: Ecotourism can provide economic incentives for conservation and can help raise public awareness about the importance of protecting polar bears and their habitat. <br>

4. Reducing waste and pollution: Reducing waste and pollution can help reduce the impact of climate change on polar bears and other species by reducing the amount of greenhouse gases released into the atmosphere. <br>

Overall, mitigation of climate change impacts on polar bears will require ongoing collaborative efforts to address the root causes of climate change and promote sustainable practices.
</blockquote>