import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import re
import json
import csv
import uuid
import torch
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 현재 파일 위치 기준으로 절대 경로 지정
# 수정 후
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR 
CSV_FILE = os.path.join(BASE_DIR, "comments.csv")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text", "timestamp"])

# PDF 파일 경로 설정 및 파일 데이터 구성
pdf_paths = [
    "정책공약집.pdf",
    "지역공약.pdf"
]

# PDF 파일 정보 - API 요청 시 반환될 정보
pdf_files_info = {
    "full": {
        "title": "더불어민주당 제20대 대통령선거 정책공약집",
        "path": "/static/pdfs/full.pdf",
        "thumbnail": "/static/images/pdf_thumbnail.jpg"
    },
    "file1": {
        "title": "삶의 터전별 공약",
        "path": "/static/pdfs/file1.pdf"
    },
    "file2": {
        "title": "대상별 공약",
        "path": "/static/pdfs/file2.pdf"
    },
    "file3": {
        "title": "1. 신경제",
        "path": "/static/pdfs/file3.pdf"
    },
    "file4": {
        "title": "2. 공정성장",
        "path": "/static/pdfs/file4.pdf"
    },
    "file5": {
        "title": "3. 민생안정",
        "path": "/static/pdfs/file5.pdf"
    },
    "file6": {
        "title": "4. 민주사회",
        "path": "/static/pdfs/file6.pdf"
    },
    "file7": {
        "title": "5. 평화안보",
        "path": "/static/pdfs/file7.pdf"
    },
    "file8": {
        "title": "소확행·명확행·SNS발표 공약",
        "path": "/static/pdfs/file8.pdf"
    }
}

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 전역 변수로 LangChain 구성요소 선언
retriever = None
llm = None
rag_chain = None

# 1. 검색 최적화를 위한 쿼리 재구성 프롬프트
query_transformation_prompt = PromptTemplate.from_template(
    """당신은 문서 검색 전문가입니다. 주어진 질문을 분석하고, 관련 문서를 효과적으로 검색하기 위한 
    최적의 검색어를 생성해주세요.
    
    원래 질문: {question}
    
    검색에 사용할 키워드나 문구만 출력하세요. 설명은 불필요합니다.
    """
)

# 2. 문서 분석 프롬프트
document_analysis_prompt = PromptTemplate.from_template(
    """당신은 정책 문서 분석 전문가입니다. 다음 문서들을 분석하고, 주어진 질문과 관련된 정책들을 찾아 정리해주세요.
    
    질문: {question}
    
    문서 내용:
    {context}
    
    위 문서에서 질문과 관련된 가장 중요한 정책/공약을 추출해주세요.
    
    # 중요 지시사항
    - 만약 질문이 이전에 언급된 특정 번호 항목(예: "3번", "4번 항목")에 대한 상세 설명을 요청하는 경우,
      다른 항목을 모두 무시하고 해당 항목만 심층적으로 분석하여 자세히 설명해주세요.
    - 항목 설명 요청일 경우, 반드시 해당 항목의 키워드("청년고용보험", "구직급여" 등)를 활용하여 문서 내용을 검색하고
      관련된 모든 세부 내용을 종합하여 설명해주세요.
    - 항목 설명 요청일 경우, 일반적인 목록 형식이 아닌 상세한 설명 형식으로 답변해주세요.
    
    # 일반 정책 질문인 경우 규칙
    1. 정책/공약은 최대 10개까지만 추출하세요. 절대로 10개를 넘기지 마세요.
    2. 추출한 정책의 수가 10개 미만이라도 무리하게 채우지 마세요.
    3. 각 정책은 핵심 내용만 간결하게 작성하세요.
    4. 각 정책은 반드시 번호를 붙여 구분하고(1. 2. 3. 등), 정책별로 한 줄씩 띄워주세요.
    5. 마지막에 출처가 되는 공약의 page와 문서명을 반드시 표시하세요.
    6. 관련 정책이 없다면 "관련 정책 정보 없음"이라고만 답하세요.
    
    답변:
    """
)

# 감정 분석 프롬프트
sentiment_analysis_prompt = PromptTemplate.from_template(
    """다음 텍스트의 감정과 의도를 분석해주세요:
    
    텍스트: {text}
    
    이 텍스트가 다음 중 하나라도 명확하게 포함하는 경우에만 "부적절"이라고 답하세요:
    1. 직접적인 비속어나 욕설
    2. 명백히 성적으로 부적절한 내용
    3. 특정 집단을 향한 혐오 표현이나 차별적 언어
    4. 직접적인 위협이나 폭력적인 내용
    5. 개인정보 요청이나 유출
    6. 명백히 불법적인 활동 유도
    7. 특정 정치인이나 개인에 대한 심한 비방이나 인신공격
    
    일반적인 질문, 정책 문의, 중립적 의견, 단순한 부정적 의견 표현 등은 "적절"로 분류하세요.
    정치적 주제나 비판적 질문이라도 예의를 갖추고 있다면 "적절"로 분류하세요.
    의도가 불분명하거나 맥락이 불충분하다면 "적절"로 분류하세요.
    
    결과(정확히 "적절" 또는 "부적절" 중 하나만 답변):
    """
)

# 로깅 설정 함수
def setup_logging():
    """로깅 시스템 설정"""
    # 로그 디렉토리 생성
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로그 파일 이름 (날짜 포함)
    log_filename = os.path.join(log_dir, f"chatbot_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    # 로거 설정
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 설정 (10MB 크기 제한, 최대 5개 백업 파일)
    file_handler = RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    
    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 감정 분석 체인 생성
def create_sentiment_analysis_chain(llm):
    sentiment_chain = (
        sentiment_analysis_prompt
        | llm
        | StrOutputParser()
    )
    return sentiment_chain

# 감정 분석 함수
async def analyze_sentiment(text, sentiment_chain):
    """텍스트의 감정과 의도를 분석하여 적절성 판단"""
    try:
        # 분석 결과 가져오기
        result = sentiment_chain.invoke({"text": text})
        
        # 로그에 분석 결과 기록
        logger.info(f"감정 분석 원본 결과: '{result}' (텍스트: '{text[:50]}...')")
        
        # 단순화된 결과 추출: "적절" 또는 "부적절" 키워드를 찾음
        is_inappropriate = False
        
        # 결과에서 마지막 50자만 검사 (최종 판단은 보통 마지막에 있음)
        last_part = result[-10:] if len(result) > 10 else result
        
        if "부적절" in last_part:
            is_inappropriate = True
        elif "적절" in last_part:
            is_inappropriate = False
        else:
            # 두 키워드가 모두 없으면 분석 실패로 간주, 기본값 사용
            logger.warning(f"감정 분석 결과에서 판단 키워드를 찾을 수 없음: {last_part}")
            is_inappropriate = False
        
        logger.info(f"부적절 여부 판단: {is_inappropriate}")
        
        return is_inappropriate, result
    except Exception as e:
        logger.error(f"감정 분석 중 오류 발생: {str(e)}")
        return False, f"오류: {str(e)}"
    
# 금지어 목록
prohibited_words = [
    "씨발", "병신", "개새끼", "지랄", "좆", "니미", "fuck", "sex", "bastard", "bitch",
    "개자식", "걸레", "창녀", "쌍놈", "쌍년", "애미", "애비", 
    "전화번호", "주민번호", "계좌번호", "신용카드", "범죄자", "쓰레기"
]

def contains_prohibited_content(text):
    """입력된 텍스트에 비속어나 개인 정보가 포함되어 있는지 확인"""
    text_lower = text.lower()
    
    # 금지어 체크
    for word in prohibited_words:
        if word.lower() in text_lower:
            return True
            
    # 이름 패턴 체크
    name_pattern = re.compile(r'[가-힣]{2,4}\s?씨')
    if name_pattern.search(text):
        return True
        
    return False

# 대화 히스토리 저장 딕셔너리
conversation_history = {}

# 대화 히스토리에 메시지 추가 함수
def add_to_history(session_id, role, content):
    """대화 히스토리에 메시지 추가"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()  # datetime 직접 사용
    })
    
    # 히스토리 크기 제한 (최근 10개 메시지만 유지)
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]

# 이전 대화 내용을 기반으로 컨텍스트 생성
def get_conversation_context(session_id):
    """이전 대화 내용을 기반으로 컨텍스트 생성 - 내부 처리용으로만 사용하고 응답에는 포함하지 않음"""
    if session_id not in conversation_history or len(conversation_history[session_id]) == 0:
        return ""
    
    # 최근 메시지 2쌍(질문+답변)만 사용
    recent_messages = conversation_history[session_id][-4:] if len(conversation_history[session_id]) >= 4 else conversation_history[session_id]
    
    # 내부 처리용 컨텍스트 생성 (프롬프트에 전달되지만 사용자에게는 표시되지 않음)
    context = ""
    for message in recent_messages:
        role_text = "USER" if message["role"] == "user" else "BOT"
        context += f"{role_text}: {message['content']}\n"
    
    return context

# 전체 멀티모달 RAG 체인 구성
def create_multimodal_rag_chain(retriever, llm):
    # 1. 쿼리 변환 체인
    query_transformer_chain = (
        {"question": RunnablePassthrough()}
        | query_transformation_prompt
        | llm
        | StrOutputParser()
    )
    
    # 2. 검색 체인 (변환된 쿼리 사용)
    def retrieve_documents(input_dict):
        original_question = input_dict["original_question"]
        optimized_query = input_dict["optimized_query"]
        
        logger.info(f"원본 질문: {original_question}")
        logger.info(f"최적화된 쿼리: {optimized_query}")
        
        # 특정 항목에 대한 상세 설명 요청인지 확인
        item_request_match = re.search(r'([0-9]+)번째|([0-9]+)번|([0-9]+)항목|([0-9]+)번 항목', original_question)
        is_item_request = bool(item_request_match)
        
        # 특정 항목 설명 요청일 경우 검색 방식 변경
        if is_item_request:
            # 이전 대화에서 항목 내용 찾기
            if "BOT:" in original_question:
                # 대화 내용에서 이전 BOT 응답 추출
                bot_response = original_question.split("BOT:")[1].strip()
                
                # 요청된 항목 번호 추출
                item_num = next(g for g in item_request_match.groups() if g is not None)
                item_num = int(item_num)
                
                # 해당 항목의 내용 추출
                items_pattern = r'([0-9]+)\.\s+(.+?)(?=\n\s*[0-9]+\.|$)'
                items = re.findall(items_pattern, bot_response, re.DOTALL)
                
                if items and 0 < item_num <= len(items):
                    # 항목 내용에서 키워드 추출
                    item_content = items[item_num-1][1].strip()
                    logger.info(f"상세 설명 요청된 항목 {item_num}번 내용: {item_content}")
                    
                    # 항목 내용을 키워드로 사용하여 검색 강화
                    extra_keywords = re.sub(r'\([^)]*\)', '', item_content)  # 괄호 내용 제거
                    keywords = ' '.join(re.findall(r'\b\w+\b', extra_keywords))
                    
                    # 최적화된 쿼리와 항목 키워드 결합
                    enhanced_query = f"{optimized_query} {keywords}"
                    logger.info(f"항목 내용 기반 강화된 쿼리: {enhanced_query}")
                    
                    # 강화된 쿼리로 검색
                    retrieved_docs = retriever.invoke(enhanced_query)
                else:
                    # 항목을 찾지 못했을 경우 기본 쿼리 사용
                    retrieved_docs = retriever.invoke(optimized_query)
            else:
                # 이전 대화 내용이 없을 경우 기본 쿼리 사용
                retrieved_docs = retriever.invoke(optimized_query)
        else:
            # 일반 질문일 경우 기본 쿼리 사용
            retrieved_docs = retriever.invoke(optimized_query)
        
        logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
        if retrieved_docs:
            logger.info(f"첫 번째 문서 일부: {retrieved_docs[0].page_content[:100]}...")
        
        # 특정 항목 설명 요청인지 여부를 컨텍스트에 포함
        return {
            "question": original_question,
            "context": "\n\n".join([doc.page_content for doc in retrieved_docs]),
            "is_item_request": is_item_request,
            "item_number": item_num if is_item_request and 'item_num' in locals() else None
        }
    
    # 3. 문서 분석 체인
    document_analysis_chain = (
        document_analysis_prompt
        | llm
        | StrOutputParser()
    )
    
    # 전체 멀티모달 체인 구성
    multimodal_chain = (
        # 1단계: 원본 질문 저장 및 쿼리 최적화
        RunnablePassthrough().with_config(run_name="Original Question")
        | {"original_question": lambda x: x, "optimized_query": query_transformer_chain}
        
        # 2단계: 최적화된 쿼리로 문서 검색
        | RunnableLambda(retrieve_documents).with_config(run_name="Document Retrieval")
        
        # 3단계: 검색된 문서 분석
        | {"question": lambda x: x["question"], 
           "analyzed_info": document_analysis_chain}
        
        # 4단계: 최종 응답 생성 (분석된 정보를 사용자 친화적 텍스트로 변환)
        | RunnableLambda(lambda x: format_response(x["question"], x["analyzed_info"]))
    )
    
    return multimodal_chain

# format_response 함수에 로그 추가
def format_response(question, analyzed_info):
    """응답을 사용자 친화적인 형식으로 변환"""
    logger.info(f"원본 질문: {question}")
    logger.info(f"분석된 정보: {analyzed_info}")
    
    # 이전 대화 내용과 새 질문 분리
    clean_question = question
    if "새로운 질문:" in question:
        clean_question = question.split("새로운 질문:")[-1].strip()
    
    # 참조 항목 정보 제거
    if "(참조 항목:" in clean_question:
        clean_question = clean_question.split("(참조 항목:")[0].strip()
    
    logger.info(f"정제된 질문: {clean_question}")
    
    # 정책 자료 다운로드 요청인지 확인
    is_download_request = any(keyword in clean_question.lower() for keyword in 
                            ["자료", "다운로드", "파일", "받기", "pdf", "정책집", "공약집"])
    
    # 정책 자료 다운로드 요청인 경우 다운로드 링크 제공
    if is_download_request:
        download_links = (
                "🤖 정책 자료를 다운로드할 수 있는 링크를 제공해 드립니다:<br><br>"
                "<div style='display: flex; gap: 10px; flex-wrap: wrap;'>"
                "<a href='/static/pdfs/full.pdf' style='padding: 10px 20px; background-color: #0078d4; color: white; text-decoration: none; border-radius: 4px; margin-bottom: 10px;' target='_blank'>정책공약집 다운로드</a>"
                "<a href='/static/pdfs/region_document.pdf' style='padding: 10px 20px; background-color: #0078d4; color: white; text-decoration: none; border-radius: 4px; margin-bottom: 10px;' target='_blank'>지역공약집 다운로드</a>"
                "</div><br>"
                "위 버튼을 클릭하여 PDF 파일을 다운로드하거나 확인하실 수 있습니다.<br><br>"
                "세부 자료를 확인하고 싶으시다면 공약정책 페이지로 이동하실 수 있습니다:<br><br>"
                "<a href='/pdf' style='padding: 10px 20px; background-color: #5cb85c; color: white; text-decoration: none; border-radius: 4px; display: inline-block;'>공약정책 페이지 이동</a><br><br>"
                "추가로 궁금하신 내용이 있으시면 말씀해 주세요."
            )
        return download_links
    
    # 응답에 "관련 정책 정보 없음"이 명시적으로 포함된 경우에만 정보 없음으로 처리
    if "관련 정책 정보 없음" in analyzed_info[-30:]:
        logger.info(f"관련 정책 정보가 없음 analyzed_info : {analyzed_info}")
        logger.info(f" analyzed_info[-30:] : {analyzed_info[-30:]}")
        return f"🤖 {clean_question}에 관한 정책 정보를 찾을 수 없습니다. 다른 질문으로 시도해 보세요."
    
    # '답변:' 이후 내용만 추출
    final_answer = analyzed_info
    if "답변:" in analyzed_info:
        final_answer = analyzed_info.split("답변:")[1].strip()
    
    # 특정 항목 설명 요청인지 확인
    is_item_request = bool(re.search(r'([0-9]+)번째|([0-9]+)번|([0-9]+)항목|([0-9]+)번 항목', clean_question))
    
    # 특정 항목 설명 요청인 경우 응답 형식 조정
    if is_item_request:
        item_match = re.search(r'([0-9]+)번째|([0-9]+)번|([0-9]+)항목|([0-9]+)번 항목', clean_question)
        item_num = next(g for g in item_match.groups() if g is not None)
        response = f"🤖 {item_num}번 항목에 대한 자세한 설명입니다.\n\n{final_answer}"
    else:
        # 일반 응답인 경우 기본 형식 사용
        response = f"🤖 {clean_question} 관련 답변드립니다.\n\n{final_answer}"
    
    # 로그에 최종 응답 기록
    logger.info(f"최종 응답: {response}...")
    
    return response

def prepare_contextual_message(user_message, session_id):
    """대화 컨텍스트와 사용자 메시지를 결합하여 최종 쿼리 생성"""
    # 1. 컨텍스트 및 참조 항목 확인
    query, context_info = process_query_with_context(user_message, session_id)
    
    # 2. 이전 대화 히스토리 확인 (최대 2개 질의응답 쌍)
    previous_context = ""
    
    if session_id in conversation_history and len(conversation_history[session_id]) > 0:
        # 최근 2쌍의 대화만 컨텍스트로 사용
        recent_msgs = []
        qa_pairs = 0
        for msg in reversed(conversation_history[session_id]):
            recent_msgs.append(msg)
            if msg["role"] == "user":
                qa_pairs += 1
                if qa_pairs >= 2:  # 최대 2개 질의응답 쌍만 사용
                    break
        
        # 다시 시간 순서대로 재정렬
        recent_msgs.reverse()
        
        # 컨텍스트 생성
        for msg in recent_msgs:
            role_text = "USER" if msg["role"] == "user" else "BOT"
            # 챗봇 응답에서 헤더 제거 (🤖 ... 관련 답변드립니다 부분)
            if role_text == "BOT" and "관련 답변드립니다" in msg["content"]:
                content = msg["content"].split("관련 답변드립니다", 1)[1].strip()
            else:
                content = msg["content"]
            
            previous_context += f"{role_text}: {content}\n"
    
    # 3. 최종 쿼리 구성
    # 항목 자세히 설명 요청인 경우 특별 처리
    if context_info and context_info.get("request_type") == "item_detail":
        final_query = (
            f"{previous_context}\n"
            f"새로운 질문: {user_message}\n"
            f"이전 질문: {context_info.get('previous_question', '')}\n"
            f"설명할 항목 번호: {context_info.get('item_number')}\n"
            f"설명할 항목 내용: {context_info.get('selected_item')}\n"
            f"요청 유형: 특정 항목에 대한 상세 설명 요청"
        )
    # 일반 참조 항목이 있을 경우 추가 정보로 제공
    elif context_info:
        final_query = (
            f"{previous_context}\n"
            f"새로운 질문: {user_message}\n"
            f"참조 항목: {context_info}"
        )
    else:
        final_query = f"{previous_context}\n새로운 질문: {user_message}"
    
    logger.info(f"최종 쿼리: {final_query[:200]}..." if len(final_query) > 200 else f"최종 쿼리: {final_query}")
    
    return final_query

# 개선된 질의 처리를 위한 함수
def process_query_with_context(query, session_id):
    """컨텍스트를 고려한 질의 처리"""
    if session_id not in conversation_history or len(conversation_history[session_id]) == 0:
        return query, None
    
    # 최근 챗봇 응답 찾기
    latest_bot_response = None
    previous_question = None
    for msg in reversed(conversation_history[session_id]):
        if msg["role"] == "assistant" and not latest_bot_response:
            latest_bot_response = msg["content"]
        elif msg["role"] == "user" and previous_question is None and latest_bot_response:
            previous_question = msg["content"]
            break
    
    if not latest_bot_response:
        return query, None
    
    # 항목 번호 추출 시도
    item_match = re.search(r'([0-9]+)번째|([0-9]+)번|([0-9]+)항목|([0-9]+)번 항목', query)
    if item_match:
        # 숫자 추출
        item_num = next(g for g in item_match.groups() if g is not None)
        item_num = int(item_num)
        logger.info(f"항목 번호 추출됨: {item_num}")
        
        # 이전 응답에서 항목 패턴 추출
        try:
            # 항목 패턴 (숫자. 내용) 찾기
            items = re.findall(r'([0-9]+)\.\s+(.+?)(?=\n\n[0-9]+\.|$)', latest_bot_response, re.DOTALL)
            if items and 0 < item_num <= len(items):
                # 찾은 항목의 내용
                item_content = items[item_num - 1][1].strip()
                logger.info(f"찾은 항목 내용: {item_content}")
                
                # 응답을 위한 컨텍스트 구성 - 이전 질문과 항목 정보 포함
                context_info = {
                    "previous_question": previous_question,
                    "selected_item": item_content,
                    "item_number": item_num,
                    "request_type": "item_detail"
                }
                
                return query, context_info
        except Exception as e:
            logger.error(f"항목 추출 오류: {str(e)}")
    
    return query, None

# init_rag_system 함수에 로그 추가
'''
def init_rag_system():
    """LangChain RAG 시스템 초기화"""
    global retriever, llm, rag_chain ,sentiment_chain

    logger.info("LangChain RAG 시스템 초기화 중...")

    # 1. PDF 문서 로드 및 텍스트 추출
    documents = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            logger.info(f"PDF 파일 로드 중: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"- {len(docs)}개 페이지 로드됨")
        else:
            logger.info(f"파일이 존재하지 않습니다: {pdf_path}")

    if not documents:
        logger.info("로드된 문서가 없습니다. 파일 경로를 확인해주세요.")
        return False
    
    logger.info(f"총 {len(documents)}개 페이지 로드됨")

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"문서를 {len(chunks)}개의 청크로 분할했습니다.")

    # 3. 임베딩 모델 설정
    logger.info("임베딩 모델 로드 중...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("임베딩 모델 로드 성공!")

    # 4. 벡터 데이터베이스 생성
    logger.info("벡터 데이터베이스 생성 중...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    logger.info("벡터 데이터베이스 생성 완료")

    # 샘플 문서 확인 (디버깅용)
    for i, doc_id in enumerate(list(vectorstore.docstore._dict.keys())[:3]):  # 처음 3개만 출력
        logger.info(f"문서 ID {doc_id}의 내용:")
        logger.info(vectorstore.docstore._dict[doc_id])
        logger.info("-" * 50)
        if i >= 2:  # 최대 3개만 출력
            break

    # 5. 검색기(Retriever) 설정 - 벡터스토어에서 직접 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 6. EXAONE 모델 로드 및 LangChain LLM 래퍼 설정
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    #model_name = "LGAI-EXAONE/EXAONE-Deep-32B-AWQ"
    #streaming = True 
    logger.info(f"{model_name} 모델 로드 중...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 양자화 설정을 BitsAndBytesConfig로 구성
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 모델 로드 - 성능 최적화 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 모델 최적화 설정 적용
    optimize_performance(model)

    # 직접 Hugging Face 파이프라인 생성 후 LangChain 래퍼 적용
    from transformers import pipeline

    # 트랜스포머 파이프라인 생성
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # LangChain HuggingFacePipeline 래퍼 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    logger.info("EXAONE 모델 로드 완료!")

    rag_chain = create_multimodal_rag_chain(retriever, llm)

    # 감정 분석 체인 생성
    sentiment_chain = create_sentiment_analysis_chain(llm)

    logger.info("LangChain RAG 시스템 초기화 완료!")
    return True
'''

# 모델 로드 및 토크나이저 설정 부분
def init_rag_system():
    """LangChain RAG 시스템 초기화"""
    global retriever, llm, rag_chain, sentiment_chain, model, tokenizer

    logger.info("LangChain RAG 시스템 초기화 중...")

    # 1-4. 벡터 데이터베이스 설정 부분 (코드 유지)...
    
    # 5. EXAONE 모델 로드 부분 수정
    model_name = "LGAI-EXAONE/EXAONE-Deep-32B-AWQ"  # 32B 모델로 변경
    logger.info(f"{model_name} 모델 로드 중...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 모델 로드 - 성능 최적화 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 또는 torch.float16
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 모델 최적화 설정 적용
    optimize_performance(model)

    # LangChain 용 커스텀 LLM 클래스 정의
    from langchain.llms.base import LLM
    from typing import Any, List, Mapping, Optional
    
    class CustomEXAONELLM(LLM):
        model: Any
        tokenizer: Any
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=800,  # 더 많은 토큰 생성
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.92,
                    repetition_penalty=1.2,  # 반복 방지
                )
            
            response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 응답이 잘리는 문제를 방지하기 위한 후처리
            response = self._clean_response(response)
            
            return response
        
        def _clean_response(self, text: str) -> str:
            """응답 텍스트 정리"""
            # 반복되는 패턴 제거 로직
            pattern = r'1\.\s+'
            matches = list(re.finditer(pattern, text))
            
            if len(matches) > 1:
                # 첫 번째 1번 항목 이후의 텍스트만 유지
                first_match_pos = matches[0].start()
                second_match_pos = matches[1].start()
                return text[:second_match_pos]
            
            return text
        
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"model_name": "Custom EXAONE Model"}
        
        @property
        def _llm_type(self) -> str:
            return "custom_exaone"
    
    # LLM 인스턴스 생성
    llm = CustomEXAONELLM(model=model, tokenizer=tokenizer)
    
    # 6. RAG 체인 및 감정 분석 체인 생성
    rag_chain = create_multimodal_rag_chain(retriever, llm)
    sentiment_chain = create_sentiment_analysis_chain(llm)

    logger.info("LangChain RAG 시스템 초기화 완료!")
    return True

def optimize_performance(model):
    """성능 최적화 설정 적용"""
    # GPU 메모리 최적화
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.current_device())

    # 모델 최적화
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        if hasattr(model.config, 'gradient_checkpointing'):
            model.config.gradient_checkpointing = False

    logger.info("성능 최적화 설정이 적용되었습니다.")

# 기본 경로 - 메인 HTML 파일 반환
@app.get('/')
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# PDF 페이지 라우트 추가
@app.get('/pdf')
async def pdf_page():
    return FileResponse(os.path.join(STATIC_DIR, "pdf.html"))

@app.get("/view-pdf")
async def view_pdf():
    # StaticFiles 미들웨어가 이미 /static 경로에 마운트되어 있으므로
    # STATIC_DIR 내부의 경로만 지정
    file_path = os.path.join(STATIC_DIR, "pdfs/region_document.pdf")
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        content_disposition_type="inline"
    )

# PDF 파일 정보 API 엔드포인트
@app.get('/api/files')
async def get_files():
    return JSONResponse(pdf_files_info)

# 댓글 모델 정의
class Comment(BaseModel):
    text: str

# 댓글 페이지 라우트 - FileResponse 사용
@app.get("/comments")
async def get_comments_page():
    return FileResponse(os.path.join(BASE_DIR, "comments.html"))

# 댓글 목록 가져오기 API
@app.get("/api/comments")
async def get_comments():
    comments = []
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                comments.append({
                    "id": row["id"],
                    "text": row["text"],
                    "timestamp": row["timestamp"]
                })
        return {"success": True, "comments": comments}
    except Exception as e:
        logger.error(f"댓글 목록 가져오기 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 댓글 저장하기 API
@app.post("/api/comments")
async def create_comment(comment: Comment):
    try:
        comment_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([comment_id, comment.text, timestamp])
        logger.info(f"새로운 댓글이 저장됨: {comment_id}")
        return {"success": True, "id": comment_id}
    except Exception as e:
        logger.error(f"댓글 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 웹 채팅 API 엔드포인트
@app.post('/api/chat')
async def chat_endpoint(request: Request):
    global rag_chain, sentiment_chain
    
    # 요청 바디 파싱
    req_data = await request.json()
    user_message = req_data.get('message', '')
    session_id = req_data.get('session_id', 'default')
    debug_mode = req_data.get('debug_mode', False)  # 디버그 모드 플래그 추가
    
    logger.info(f"세션 {session_id}에서 새로운 메시지 수신: {user_message[:30]}..." if len(user_message) > 30 else user_message)
    
    if not user_message:
        logger.warning("빈 메시지가 전송됨")
        return JSONResponse({"response": "메시지를 입력해주세요."})
    
    try:
        sentiment_debug_info = {}  # 감정 분석 디버그 정보
        
        # 1단계: 기본 금지어 필터링
        if contains_prohibited_content(user_message):
            logger.warning(f"금지어 필터링 - 부적절한 내용 감지: {user_message[:30]}...")
            response = "죄송합니다. 부적절한 언어나 개인정보가 포함된 질문에는 답변할 수 없습니다."
            sentiment_debug_info["filter_type"] = "prohibited_words"
            
            if debug_mode:
                return JSONResponse({
                    "response": response,
                    "debug_info": sentiment_debug_info
                })
            return JSONResponse({"response": response})
        
        # 2단계: 감정 분석을 통한 부정적 어휘 필터링
        is_inappropriate, analysis_result = await analyze_sentiment(user_message, sentiment_chain)
        
        # 디버그 정보 저장
        sentiment_debug_info = {
            "analysis_result": analysis_result,
            "is_inappropriate": is_inappropriate
        }
        
        if is_inappropriate:
            logger.warning(f"감정 분석 필터링 - 부정적 내용 감지: {user_message[:30]}...")
            response = "죄송합니다. 부적절하거나 부정적인 내용이 포함된 질문에는 답변할 수 없습니다."
            sentiment_debug_info["filter_type"] = "sentiment_analysis"
            
            if debug_mode:
                return JSONResponse({
                    "response": response,
                    "debug_info": sentiment_debug_info
                })
            return JSONResponse({"response": response})
        
        # 1. 컨텍스트 및 질의 준비 (이전 대화 + 새 질문)
        contextual_query = prepare_contextual_message(user_message, session_id)
        
        # 2. RAG 체인으로 응답 생성
        logger.info("RAG 체인으로 응답 생성 중...")
        response = rag_chain.invoke(contextual_query)
        
        # 3. 대화 히스토리에 저장 (원본 질문과 응답)
        add_to_history(session_id, "user", user_message)
        add_to_history(session_id, "assistant", response)
        
        return JSONResponse({
            "response": response,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"요청 처리 중 오류 발생: {str(e)}", exc_info=True)
        return JSONResponse({
            "response": "죄송합니다. 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "session_id": session_id,
            "error": str(e) if debug_mode else None
        })

# 서버 초기화 및 실행을 위한 이벤트
@app.on_event("startup")
async def startup_event():
    global logger
    
    # 로깅 시스템 설정
    logger = setup_logging()
    logger.info("==== 웹 챗봇 서버 시작 ====")
    
    # 모델 및 RAG 시스템 초기화
    logger.info("LangChain RAG 시스템 초기화 시작")
    init_success = init_rag_system()

    if not init_success:
        logger.error("초기화 실패. 서버를 시작할 수 없습니다.")
        import sys
        sys.exit(1)
    else:
        logger.info("서버 시작 준비 완료! 웹 챗봇 서버를 실행합니다.")
  

# 서버 실행
if __name__ == "__main__":
    # 80번 포트 사용 (root 권한 필요)
    #uvicorn.run("chatbot_web:app", host="0.0.0.0", port=80, reload=True)
    
    # 개발용으로는 5000번 포트 사용
    uvicorn.run("chatbot_web:app", host="0.0.0.0", port=5000,http="auto", reload=True)
    # HTTPS로 실행
    #uvicorn.run(
    #    "chatbot_web:app", 
    #    host="0.0.0.0", 
    #    port=5000, 
    #    ssl_keyfile=os.path.join(BASE_DIR, "ca/private.pem"),  # 개인 키 경로 
    #    ssl_certfile=os.path.join(BASE_DIR, "ca/private_ct.pem"),  # 인증서 경로
    #    reload=True
    #)
