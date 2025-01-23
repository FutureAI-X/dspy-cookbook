import dspy

lm = dspy.LM('ollama_chat/qwen2.5:0.5b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

response1 = lm("你是谁", temperature=0.7)  # => ['This is a test!']
response2 = lm(messages=[{"role": "user", "content": "你是谁"}])  # => ['This is a test!']

print(response1)
print(response2)