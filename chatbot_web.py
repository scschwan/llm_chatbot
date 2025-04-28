
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import re
import csv
import uuid
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

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

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PDF 파일 경로 설정
pdf_paths = [
    "정책공약집.pdf",
    "지역공약.pdf"
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "comments.csv")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text", "timestamp"])

# 정적 파일 및 템플릿 디렉토리 경로 설정
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

# PDF 파일 정보 - API 요청 시 반환될 정보
pdf_files_info = {
    "full": {
        "title": "더불어민주당 제20대 대통령선거 정책공약집",
        "path": "static/pdfs/full.pdf",
        "thumbnail": "static/images/pdf_thumbnail.jpg"
    },
    "file1": {
        "title": "삶의 터전별 공약",
        "path": "static/pdfs/file1.pdf"
    },
    "file2": {
        "title": "대상별 공약",
        "path": "static/pdfs/file2.pdf"
    },
    "file3": {
        "title": "1. 신경제",
        "path": "static/pdfs/file3.pdf"
    },
    "file4": {
        "title": "2. 공정성장",
        "path": "static/pdfs/file4.pdf"
    },
    "file5": {
        "title": "3. 민생안정",
        "path": "static/pdfs/file5.pdf"
    },
    "file6": {
        "title": "4. 민주사회",
        "path": "static/pdfs/file6.pdf"
    },
    "file7": {
        "title": "5. 평화안보",
        "path": "static/pdfs/file7.pdf"
    },
    "file8": {
        "title": "소확행·명확행·SNS발표 공약",
        "path": "static/pdfs/file8.pdf"
    }
}


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

# 감정 분석 체인 생성
def create_sentiment_analysis_chain(llm):
    sentiment_chain = (
        sentiment_analysis_prompt
        | llm
        | StrOutputParser()
    )
    return sentiment_chain

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

# 전역 변수로 모델과 벡터 스토어 선언
model = None
tokenizer = None
vectorstore = None

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


def init_rag_system():
    """RAG 시스템 초기화"""
    global model, tokenizer, vectorstore

    logger.info("RAG 시스템 초기화 중...")

    # 1. PDF 문서 로드 및 텍스트 추출
    documents = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        else:
            logger.info(f"파일이 존재하지 않습니다: {pdf_path}")

    if not documents:
        logger.info("로드된 문서가 없습니다. 파일 경로를 확인해주세요.")
        return False

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

    for doc_id in list(vectorstore.docstore._dict.keys())[:3]:  # 처음 3개만 출력해 로그 크기 제한
        logger.info(f"문서 ID {doc_id}의 내용 (샘플):")
        logger.info(vectorstore.docstore._dict[doc_id])
        logger.info("-" * 50)

    # 5. EXAONE 모델 로드
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
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
    logger.info("EXAONE 모델 로드 완료!")

    # 성능 최적화 설정 적용
    optimize_performance()

    return True

def optimize_performance():
    """성능 최적화 설정 적용"""
    global model

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

def retrieve_context(query, k):
    """쿼리와 관련된 문서를 검색하여 컨텍스트를 생성합니다."""
    global vectorstore
    relevant_docs = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

def generate_answer(prompt):
    """프롬프트에 대한 응답을 생성합니다."""
    global model, tokenizer

    # 입력 인코딩 - attention_mask 명시적 설정
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # 효율적인 생성 설정
    generation_config = {
        "max_new_tokens": 500,
        # "do_sample": True,
        # "temperature": temperature,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # 토큰 생성
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # attention_mask 명시적 전달
            **generation_config
        )

    # 응답 디코딩
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def post_process_answer(answer):
    """응답을 후처리하여 일관된 형식으로 정제합니다."""
    # 불필요한 마크다운 제거
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
    
    # "다음과 같습니다" 패턴 제거
    answer = re.sub(r'^(다음과 같습니다|관련 공약은 다음과 같습니다|다음과 같은 내용이 있습니다|다음과 같은 공약이 있습니다|다음을 참고하세요)[\\.:]?\\s*', '', answer)
    
    # "~입니다"로 시작하는 패턴 제거
    answer = re.sub(r'^[^\\.]*입니다[\\.:]?\\s*', '', answer)
    
    # 모든 콜론 뒤에 줄바꿈 추가
    answer = re.sub(r':\\s*', ':\n', answer)
    
    # 번호 리스트 형식 정리
    answer = re.sub(r'(\\d+)\\.\\s+', r'\n\\1. ', answer)
    
    # 불필요한 연속된 줄바꿈 정리
    answer = re.sub(r'(<br>){3,}', '<br><br>', answer)
    
    # 첫 번째 줄이 비어 있으면 제거
    answer = re.sub(r'^(<br>)', '', answer)
    
    # 응답이 비어 있는 경우 처리
    if not answer.strip():
        answer = "관련 정보를 찾을 수 없습니다."
        
    return answer.strip()

def answer_with_rag(query):
    """RAG로 컨텍스트를 검색하고 일관된 형식으로 응답을 생성합니다."""
    context = retrieve_context(query, k=5)

    # 개선된 프롬프트 구성 - 명확한 응답 형식 지정
    prompt = f"""제공한 정보를 바탕으로 사용자 질문에 답하세요.
              문서 내용에 없는 정보는 추측하지 말고, 정보가 부족하면 솔직히 모른다고 말하세요.
              
              ### 응답 형식 ###
              답변은 다음과 같은 일관된 형식으로 작성하세요:
              1. 질문 주제와 관련된 공약 또는 정책을 항목별로 나눠 작성합니다.
              2. 모든 요점 앞에는 번호나 기호(예: 1. 2. 3. 또는 - - -)를 붙여 항목별로 구분합니다.
              3. 각 항목은 다음 줄에 작성하여 가독성을 높입니다.
              4. "다음과 같습니다", "~ 입니다", "다음을 참고하세요" 등의 표현으로 시작하지 마세요.
              5. 각 항목은 짧고 명확한 문장으로 작성합니다.
              
              ### 참고 정보: ### 
              {context}
              
              ### 사용자 질문 ###
              {query}와 관련된 공약만 문서에 있는 그대로 답변해주세요.
              
              ### 답변 ###"""

    # 응답 생성
    raw_answer = generate_answer(prompt)
    
    # 후처리를 통한 응답 정제
    answer = post_process_answer(raw_answer)
    return answer

def create_exaone_pipeline(model, tokenizer):
    """EXAONE 파이프라인 생성 유틸리티 함수"""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get('/')
async def get_index():
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse({"message": "index.html 파일을 찾을 수 없습니다"}, status_code=404)

# PDF 페이지 라우트
@app.get('/pdf')
async def pdf_page():
    return FileResponse(os.path.join(templates_dir, "pdf.html"))

# Comments 페이지 라우트
@app.get('/comments')
async def comments_page():
    return FileResponse(os.path.join(templates_dir, "comments.html"))

# PDF 파일 정보 API 엔드포인트
@app.get('/api/files')
async def get_files():
    return JSONResponse(pdf_files_info)


@app.get("/view-pdf")
async def view_pdf():
    # StaticFiles 미들웨어가 이미 /static 경로에 마운트되어 있으므로
    # STATIC_DIR 내부의 경로만 지정
    file_path = os.path.join(static_dir, "pdfs/region_document.pdf")
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        content_disposition_type="inline"
    )

# PDF 다운로드 라우트
@app.get('/download/{file_id}')
async def download_pdf(file_id: str):
    if file_id in pdf_files_info:
        file_path = pdf_files_info[file_id]["path"]
        return FileResponse(
            path=os.path.join(static_dir, file_path),
            filename=f"{pdf_files_info[file_id]['title']}.pdf",
            media_type="application/pdf"
        )
    return JSONResponse({"error": "파일을 찾을 수 없습니다."}, status_code=404)

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


# 댓글 모델 정의
class Comment(BaseModel):
    text: str

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



# 웹 버전 채팅 엔드포인트 
@app.post('/api/chat')
async def web_chat(request: Request):
    # 웹 요청 파싱
    req = await request.json()
    
    
    session_id = req.get('session_id', 'default')

    # 사용자 메시지 추출
    user_query = req.get('message', '')

    logger.info(f"세션 {session_id}에서 새로운 메시지 수신: {user_query[:30]}..." if len(user_query) > 30 else user_query)
    
    if not user_query:
        return JSONResponse({
            "response": "메시지가 없습니다. 질문을 입력해주세요."
        })
    
    try:        
        # 1단계: 기본 금지어 필터링
        if contains_prohibited_content(user_query):
            logger.warning(f"금지어 필터링 - 부적절한 내용 감지: {user_query[:30]}...")
            response = "죄송합니다. 부적절한 언어나 개인정보가 포함된 질문에는 답변할 수 없습니다."

            return JSONResponse({
                "response": response
            }, headers={"Content-Type": "application/json; charset=utf-8"})

        
      

           
        answer = answer_with_rag(user_query)
        
        logger.info(f"질문: {user_query}")
        logger.info(f"응답: {answer}")

        # 웹 클라이언트 응답 형식 (HTML 태그가 해석되도록 safe=False 설정)
        return JSONResponse({
            "response": answer
        }, headers={"Content-Type": "application/json; charset=utf-8"})
        
    except Exception as e:
        logger.info(f"오류 발생: {str(e)}")
        return JSONResponse({
            "response": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)

# 서버 초기화 및 실행을 위한 이벤트
@app.on_event("startup")
async def startup_event():
    # 로깅 시스템 설정
    logger = setup_logging()
    logger.info("==== 웹 챗봇 서버 시작 ====")
    
    # 모델 및 RAG 시스템 초기화
    logger.info("LangChain RAG 시스템 초기화 시작")

    # 모델 및 RAG 시스템 초기화
    init_success = init_rag_system()

    if not init_success:
        logger.error("초기화 실패. 서버를 시작할 수 없습니다.")
        import sys
        sys.exit(1)
    else:
        logger.info("서버 시작 준비 완료! 서버를 실행합니다.")
        
# 서버 실행
if __name__ == "__main__":
    uvicorn.run("chatbot_web:app", host="0.0.0.0", port=5000)