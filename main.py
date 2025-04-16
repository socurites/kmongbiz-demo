import os
# from dotenv import load_dotenv # .env 로딩 제거
import pandas as pd
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, END
import requests
import json
from tavily import TavilyClient
import streamlit as st

# 환경 변수 로드 제거
# load_dotenv()

# OpenAI 클라이언트 초기화
# client = OpenAI()

# Tavily 클라이언트 초기화 (st.secrets 사용)
tavily_api_key = st.secrets.get("TAVILY_API_KEY")
if not tavily_api_key:
    st.error("Tavily API 키가 설정되지 않았습니다. Streamlit Cloud Secrets를 확인하세요.")
    st.stop()
tavily_client = TavilyClient(api_key=tavily_api_key)

# Tavily API를 사용한 웹 검색
def search_web(query):
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_answer="advanced",
        max_results=10,
    )

    return response

# 카테고리 데이터 로드
def load_category_data():
    return pd.read_csv('kmong-category-3depth.tsv', sep='\t')

# 기업 정보 조회 함수
def get_company_info(company_name, business_number):
    # 웹 검색을 통한 기업 정보 조회
    search_query = f'"{company_name}\t{business_number}"에 대한 기업의 최대한 상세한 설명(핵심 사업,제품, 주력상품,고객층, 미션, 비전, 포트폴리오, 최근 중요 홍보기사내용)과 홈페이지 URL. 설명은 description에, url은 urk 키로 하여 json format으로 정리할 것.'
    search_results = search_web(search_query)
    
    try:
        # answer에서 JSON 형식의 데이터 추출
        answer_json = json.loads(search_results.get('answer', '{}'))
        description = answer_json.get('description', '')
        website = answer_json.get('url', '정보없음')
    except json.JSONDecodeError:
        description = search_results.get('answer', '')
        website = '정보없음'
    
    # 검색 결과에서 기업 정보 추출
    company_info = {
        "company_name": company_name,
        "business_number": business_number,
        "website": website,
        "description": description
    }
    
    return company_info

# OpenAI API를 사용하여 추천 생성
# def get_openai_recommendation(prompt_text):
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "당신은 크몽 카테고리를 추천하는 도우미입니다. 모든 응답은 반드시 한글로 작성해주세요."},
#             {"role": "user", "content": prompt_text}
#         ]
#     )
#     return response.choices[0].message.content

# LangGraph 노드 정의
def company_info_node(state):
    company_name, business_number = state["user_input"].split(",")
    company_info = get_company_info(company_name, business_number)
    return {"company_info": company_info}

# def category_recommendation_node(state):
#     company_info = state["company_info"]
#     categories = load_category_data()
    
#     # 웹 검색 결과를 포함한 프롬프트 생성
#     additional_info = "\n".join([
#         f"- {info['title']}\n  {info['snippet']}\n  URL: {info['url']}"
#         for info in company_info['additional_info']
#     ])
    
#     prompt = f"""
#     - 첨부: 크몽 카테고리 정보 (kmong-category-3depth.tsv 파일)

#     - 기업정보 조회할 것: {company_info}

#     - AI 요약:
#     {company_info['ai_summary']}

#     - 웹 검색 결과:
#     {additional_info}

#     - 해당 기업은 크몽 Biz 기업 구매자야. 기업의 정보를 바탕으로 구매 가능성이 높은 크몽의 카테고리를 3차 카테고리를 추천할 것. 근거 필요

#     - 기업정보 조회가 되지 않는 경우 "정보없음" 출력할 것

#     - 응답은 반드시 한글로 작성해주세요.
#     """
    
#     result = get_openai_recommendation(prompt)
#     return {"recommendation": result}

# 워크플로우 정의
workflow = Graph()

# workflow.add_node("company_info", company_info_node)
# workflow.add_node("category_recommendation", category_recommendation_node)

# workflow.add_edge("company_info", "category_recommendation")
# workflow.add_edge("category_recommendation", END)

workflow.add_node("company_info", company_info_node)

workflow.add_edge("company_info", END)

workflow.set_entry_point("company_info")

# 실행 함수
def run_workflow(user_input):
    app = workflow.compile()
    result = app.invoke({"user_input": user_input})
    #return result["recommendation"]
    return result["company_info"]

# Streamlit 웹 인터페이스
def main():
    st.title("기업 정보 검색 시스템")
    
    with st.form(key='search_form'):
        # 사용자 입력 받기
        user_input = st.text_input("기업명과 사업자번호를 탭으로 구분하여 입력하세요 (예: 삼성전자\t1234567890)")
        submitted = st.form_submit_button("검색")

    if submitted and user_input:
        try:
            company_name, business_number = user_input.split('\t')
            with st.spinner("검색 중..."):
                # 기업 정보 조회
                company_info = get_company_info(company_name, business_number)
                
                # 결과 표시
                st.subheader("검색 결과")
                
                # 기업 기본 정보
                st.markdown(f"### 기업정보: {company_info['company_name']} ({company_info['business_number']})")
                
                # 홈페이지
                st.markdown(f"### 홈페이지: {company_info['website']}")
                
                # 기업 설명
                st.markdown("### 기업 설명")
                if company_info['description']:
                    st.markdown(company_info['description'].replace('\n', '<br>'), unsafe_allow_html=True)
                else:
                    st.markdown("정보가 없습니다.")
        except ValueError:
            st.error("입력 형식이 올바르지 않습니다. 기업명과 사업자번호를 탭으로 구분하여 입력해주세요.")
    elif submitted and not user_input:
        st.warning("기업명과 사업자번호를 입력해주세요.")

if __name__ == "__main__":
    main() 