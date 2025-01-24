"""数据处理DEMO"""
import dspy

# 1、简单案例
qa_pair = dspy.Example(question='问题', answer='答案')
print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)