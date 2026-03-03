from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class ResearchState(TypedDict):
    topic: str
    questions: str
    answers: str
    summary:str
    revised_summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_questions(state: ResearchState):
    try:
        print(f"\n📋 Generating questions about: {state['topic']}")
        response = llm.invoke([
        SystemMessage(content="""You are a senior research analyst with decades of experience
            in research across wide variety of industries about technology, strategy and digital transformation. 
            You ask incisive questions that uncover business impact, technical challenges, risk and opportunities,
            laying out strategic implications."""),
        # HumanMessage(content=f"Generate 3 concise research questions about: {state['topic']}") GENERIC
        HumanMessage(content=f"""Generate 3 insightful research questions about: {state['topic']}
                     
            Each question should explore one of these angles:
                - Business and strategic impact
                - Technical challenges and considerations  
                - Implementation and best practices
                     
            Format each question on a new line numbered 1, 2, 3.""") 
            
        ])
        return {"questions": response.content}
    except Exception as e:
        print(f"Error generating questions: {e}")
        return {"questions": "Error occurred while generating questions"}

def answer_questions(state: ResearchState):
    try:
        print("\n💡 Answering questions...")
        response = llm.invoke([
            # SystemMessage(content="You are a knowledgeable research assistant."),
            SystemMessage(content="""You are a seasoned technology practitioner with decades of experience
            working for Fortune 500 companies across industry domains. You provide thoughtful yet objective answers 
            to the questions asked. Where specific data is unavailable clearly state assumptions and distinguish between 
            known facts and emerging trends or ideas for innovation."""),
            
            HumanMessage(content=f"""Answer these questions in a substantive yet focused manner:{state['questions']}
            
            Each answer should be minimum 3-4 lines and no more than 20 lines,with following guidelines:
            1) Identify traps and pitfalls to avoid
            2) List concrete industry example both success and failure where possible
            3) Identify trends and emerging best practices along with guardrails and risk mitigation strategies
                         
            Each answer should be numbered corresponding to the question number and formatted on a new line.""")
        ])
        return {"answers": response.content}
    except Exception as e:
        print(f"Error answering questions: {e}")
        return {"answers": "Error occurred while answering questions"}
    
def summarize(state: ResearchState):
    try:
        print("\n📝 Summarizing answers...")
        response = llm.invoke([
            SystemMessage(content="""You are an experienced technology writer. You communicate complex technical concepts clearly to both 
            technical and non-technical audiences without fluff. Your summary should cover core points from the answers and aligned with questions.
            You should avoid using your bias,judgement or opinion at all COSTS and only summarize the content provided in the answers, 
            without adding any new information or insights."""),
            HumanMessage(content=f"""Write a brief summary report based on:
            Topic: {state['topic']} 
            Questions:{state['questions']} 
            and Answers:{state['answers']}

            The summary should 
            1) not be more than 15 lines but should be at least 4-5 lines
            2) encompass answers to all the question
            3) have a conclusion or some inference that aligns with the topic and in line with the answers
            4) use simple language and avoid technical jargon where possible, while still being accurate.""")
        ])
        return {"summary": response.content}
    except Exception as e:
        print(f"Error summarizing: {e}")
        return {"summary": "Error occurred while summarizing"}
    
def critique(state: ResearchState):
    try:
        print("\n📝 Objective criticism and the revised summary...")
        response = llm.invoke([
            SystemMessage(content="""You are a hard nosed, experienced industry veteran and critical thinker 
            with decades of experience across technology and business transformation. 
            You have seen many initiatives succeed and fail. You provide brutally honest, 
            objective critique without personal bias. You back every criticism with reasoning 
            and always suggest concrete improvements."""),

            HumanMessage(content=f"""Review and critique the following research report on {state['topic']}.

            Topic: {state['topic']}
            Questions: {state['questions']}
            Answers: {state['answers']}
            Summary: {state['summary']}

            Your critique should:
            1) Identify gaps, weaknesses or missing perspectives in the answers
            2) Challenge any assumptions that are not well supported
            3) Identify what was done well and should be retained
            4) Provide a revised and improved FINAL REPORT that addresses the gaps

            Format your response as:
            CRITIQUE:
            [your critique here]

            FINAL REPORT:
            [revised report here]""")])
        return {"revised_summary": response.content}
    except Exception as e:
        print(f"Error critiquing: {e}")
        return {"revised_summary": "Error occurred while critiquing and revising summary"}
    
workflow = StateGraph(ResearchState)

workflow.add_node("generate_questions", generate_questions)
workflow.add_node("answer_questions", answer_questions)
workflow.add_node("summarize", summarize)
workflow.add_node("critique", critique)

workflow.set_entry_point("generate_questions")
workflow.add_edge("generate_questions", "answer_questions")
workflow.add_edge("answer_questions", "summarize")
workflow.add_edge("summarize", "critique")
workflow.add_edge("critique", END)

pipeline = workflow.compile()

topic = input("\nEnter a research topic: ")

result = pipeline.invoke({"topic": topic})

print("\n" + "="*50)
print("RESEARCH QUESTIONS")
print("="*50)
print(result["questions"])

print("\n" + "="*50)
print("ANSWERS")
print("="*50)
print(result["answers"])

# print("\n" + "="*50)
# print("INITIAL SUMMARY")
# print("="*50)
# print(result["summary"])

with open("./outcomes/report.txt", "w") as f:
    f.write(result["summary"])
# print("FINAL REPORT")
# print("###"*50)
# print(result["revised_summary"])

with open("./outcomes/final_report.txt", "w") as f:
    f.write(result["revised_summary"])

print("\n" + "="*50)
print("✅ Reports saved to:")
print("   ./outcomes/report.txt")
print("   ./outcomes/final_report.txt")
print("="*50)