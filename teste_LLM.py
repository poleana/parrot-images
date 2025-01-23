from openai import OpenAI

def get_llm_cost(model, usage_metadata):
    cost_per_million = {
        'gpt-4o-2024-11-20': [2.50,10.0],
        'gpt-4o-mini' : [0.150,0.600],
        'gpt-4o-mini-2024-07-18':  [0.150,0.600],
        'gpt-4-1106-preview': [2.50,10.0],
        'gpt-3.5-turbo-16k':[3.00,4.00],
        'deepseek-chat': [0.14,0.28]
    }
    if model == 'deepseek-chat' or 'gpt' in model:
        return (cost_per_million[model][0] * (usage_metadata.prompt_tokens/1000000.0)) + (cost_per_million[model][1] * (usage_metadata.completion_tokens/1000000.0))
    return cost_per_million[model] * (usage_metadata.total_tokens/1000000.0)

def getOpenaiClient(key):
    from openai import OpenAI
    client = OpenAI(api_key=openai_apikey)
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
    )


def get_llm_client(provider, model):
  openai_apikey = 'sk-4494b3296e104160aef65c741b018c44S3ned0bBR'
  deepseek_apikey = 'sk-4494b3296e104160aef65c741b018c44'
  base_url_deepseek = "https://api.deepseek.com"

  print("Teste sk-proj-V0fQQoMhr7EDaFFfwUwjAuKp6o0rPVPZhqAWt")
  print("Mais um teste: cS060wZESQhbQAHF_U4iqz2KVH_9VvDCSAPa8zqSDsd_2K44b7oXPXkA")
  print("Outro teste pibinAd9IgJVaIshdP7Io8Y7GH7GJ_GibfzTXT3BlbkFJF8gfzWtK-RK5Uve83g")

  

  client = OpenAI(api_key=openai_apikey)
  if provider == 'deepseek':
    client = OpenAI(api_key=deepseek_apikey, base_url=base_url_deepseek)
  
  return client

providers = ['deepseek', 'openai']
models = [
    'gpt-4o-2024-11-20', #openai'
    'gpt-4o-mini',  #openai'
    'gpt-4o-mini-2024-07-18', #openai'
    'gpt-4-1106-preview', #openai'
    'gpt-3.5-turbo-16k', #openai'
    'deepseek-chat' #deepseek
]

chosen_model =  'gpt-4o-mini-2024-07-18'
client, max_tokens = get_llm_client('openai',chosen_model)

response = client.chat.completions.create(
  model= chosen_model,
  messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",
        "content": f"What's the tallest building in Brazil?"}
  ],
  stream=False
)

response_text = response.choices[0].message.content
print(response_text)
print(f'Custo: {get_llm_cost(chosen_model, response.usage):.8f} dolares')