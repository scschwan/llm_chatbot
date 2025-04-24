import os
import re
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
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
    
    위 문서에서 질문과 관련된 가장 중요한 정책/공약을 최대 10개까지 추출하여 1,000자 이내로 요약해주세요.
    각 정책은 핵심 내용만 간결하게 작성하세요.
    또한 마지막에 출처가 되는 공약의 page 와 문서명을 반드시 답변해주세요.
    관련 정책이 없다면 "관련 정책 정보 없음"이라고 답하세요.
    """
)

'''
위 문서에서 질문과 관련된 정책/공약만 추출하세요. 
각 정책/공약의 핵심 내용과 위치한 문서 부분을 명시하세요.
관련 정책이 없다면 "관련 정책 정보 없음"이라고 답하세요.
'''


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
        
        print(f"원본 질문: {original_question}")
        print(f"최적화된 쿼리: {optimized_query}")
        
        # 최적화된 쿼리로 검색
        retrieved_docs = retriever.invoke(optimized_query)
        
        print(f"검색된 문서 수: {len(retrieved_docs)}")
        if retrieved_docs:
            print(f"첫 번째 문서 일부: {retrieved_docs[0].page_content[:100]}...")
        
        return {
            "question": original_question,
            "context": "\n\n".join([doc.page_content for doc in retrieved_docs])
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
    print(f"질문: {question}")
    print(f"분석된 정보: {analyzed_info}")
    
    # analyzed_info에서 정책 정보 추출
    policies = []
    
    # 간단한 텍스트 처리로 정책 항목 추출
    info_lines = analyzed_info.split('\n')
    current_policy = None
    
    for line in info_lines:
        line = line.strip()
        if not line:
            continue
            
        # 새 정책 항목 시작으로 보이는 패턴
        if re.match(r'^[0-9]+\.', line) or re.match(r'^•', line) or re.match(r'^-', line):
            if current_policy:
                policies.append(current_policy)
            current_policy = line
        elif current_policy:
            current_policy += " " + line
    
    # 마지막 정책 추가
    if current_policy:
        policies.append(current_policy)
    
    print(f"추출된 정책 수: {len(policies)}")
    if len(policies) > 0:
        print(f"첫 번째 정책: {policies[0]}")
    
    else:
        print("관련 정책 정보가 없음")
        return f"🤖 {question} 관련 답변드립니다.\n\n죄송합니다. 요청하신 '{question}'에 관한 정책 정보를 찾을 수 없습니다. 다른 질문으로 시도해 보세요."
    
    # 응답 구성
    response = f"🤖 {question} 관련 답변드립니다.\n\n"
    for i, policy in enumerate(policies, 1):  # 최대 3개 정책만 표시
        response += f"{i}. {policy}\n\n"
    
    return response

# init_rag_system 함수에 로그 추가
def init_rag_system():
    """LangChain RAG 시스템 초기화"""
    global retriever, llm, rag_chain

    print("LangChain RAG 시스템 초기화 중...")

    # 1. PDF 문서 로드 및 텍스트 추출
    documents = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            print(f"PDF 파일 로드 중: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"- {len(docs)}개 페이지 로드됨")
        else:
            print(f"파일이 존재하지 않습니다: {pdf_path}")

    if not documents:
        print("로드된 문서가 없습니다. 파일 경로를 확인해주세요.")
        return False
    
    print(f"총 {len(documents)}개 페이지 로드됨")

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"문서를 {len(chunks)}개의 청크로 분할했습니다.")

    # 3. 임베딩 모델 설정
    print("임베딩 모델 로드 중...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("임베딩 모델 로드 성공!")

    # 4. 벡터 데이터베이스 생성
    print("벡터 데이터베이스 생성 중...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print("벡터 데이터베이스 생성 완료")

    # 샘플 문서 확인 (디버깅용)
    for i, doc_id in enumerate(list(vectorstore.docstore._dict.keys())[:3]):  # 처음 3개만 출력
        print(f"문서 ID {doc_id}의 내용:")
        print(vectorstore.docstore._dict[doc_id])
        print("-" * 50)
        if i >= 2:  # 최대 3개만 출력
            break

    # 5. 검색기(Retriever) 설정 - 벡터스토어에서 직접 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 6. EXAONE 모델 로드 및 LangChain LLM 래퍼 설정
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    print(f"{model_name} 모델 로드 중...")

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
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # LangChain HuggingFacePipeline 래퍼 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print("EXAONE 모델 로드 완료!")

    rag_chain = create_multimodal_rag_chain(retriever, llm)

    print("LangChain RAG 시스템 초기화 완료!")
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

    print("성능 최적화 설정이 적용되었습니다.")

# 기본 경로 - 메인 HTML 파일 반환
@app.get('/')
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# PDF 페이지 라우트 추가
@app.get('/pdf')
async def pdf_page():
    return FileResponse(os.path.join(STATIC_DIR, "pdf.html"))

# PDF 파일 정보 API 엔드포인트
@app.get('/api/files')
async def get_files():
    return JSONResponse(pdf_files_info)

# 웹 채팅 API 엔드포인트
@app.post('/api/chat')
async def chat_endpoint(request: Request):
    global rag_chain
    
    # 요청 바디 파싱
    req_data = await request.json()
    user_message = req_data.get('message', '')
    
    if not user_message:
        return JSONResponse({"response": "메시지를 입력해주세요."})
    
    try:
        # RAG 체인으로 응답 생성
        response = rag_chain.invoke(user_message)
        
        # 클라이언트 응답 형식
        return JSONResponse({
            "response": response
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse({
            "response": "죄송합니다. 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        })

# 서버 초기화 및 실행을 위한 이벤트
@app.on_event("startup")
async def startup_event():
    # 모델 및 RAG 시스템 초기화
    init_success = init_rag_system()

    if not init_success:
        print("초기화 실패. 서버를 시작할 수 없습니다.")
        import sys
        sys.exit(1)
    else:
        print("서버 시작 준비 완료! 웹 챗봇 서버를 실행합니다.")

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
