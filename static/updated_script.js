document.addEventListener('DOMContentLoaded', () => {
    // 기존 코드에서 필요한 DOM 요소
    const messagesContainer = document.getElementById('messages-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    
    // 새로운 요소들
    const landingPage = document.getElementById('landing-page');
    const chatInterface = document.getElementById('chat-interface');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    // 상태 변수
    let isWaitingForResponse = false;

    // API 기본 URL (기존 코드에서 유지)
    const API_URL = '/api/chat';

    // 텍스트 영역 자동 높이 조절 (기존 코드에서 유지하되 수정)
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight > 150 ? 150 : messageInput.scrollHeight) + 'px';
    });

    // 엔터 키로 메시지 전송 (Shift+Enter는 줄바꿈) - 기존 코드에서 유지
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 전송 버튼 클릭으로 메시지 전송 - 기존 코드에서 유지
    sendButton.addEventListener('click', sendMessage);

    // 메시지 전송 함수 - 기존 코드에서 유지하되 수정
    function sendMessage() {
        const message = messageInput.value.trim();
        
        if (message && !isWaitingForResponse) {
            // 인트로 화면이 표시 중이라면 채팅 화면으로 전환
            if (!landingPage.classList.contains('hidden')) {
                landingPage.classList.add('hidden');
                chatInterface.classList.remove('hidden');
            }
            
            // 사용자 메시지 추가
            addMessage(message, 'user');
            
            // 입력 필드 초기화
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // 로딩 표시 추가
            const loadingElement = document.createElement('div');
            loadingElement.className = 'loading';
            loadingElement.innerHTML = `
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            `;
            messagesContainer.appendChild(loadingElement);
            scrollToBottom();
            
            // 입력 상태 업데이트
            isWaitingForResponse = true;
            sendButton.disabled = true;
            
            // 서버와 통신 (기존 함수 유지)
            sendToServer(message, loadingElement);
        }
    }

    // 메시지 추가 함수 - 기존 코드 유지하되 HTML 지원 추가
    function addMessage(text, sender, isError = false) {
        const messageElement = document.createElement('div');
        
        if (isError) {
            messageElement.className = 'message error-message';
        } else {
            messageElement.className = `message ${sender}-message`;
        }
        
        // HTML 지원을 위해 textContent 대신 innerHTML 사용
        messageElement.innerHTML = text;
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }

    // 맨 아래로 스크롤 - 기존 코드 유지
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // 서버로 메시지 전송 및 응답 처리 - 기존 코드 유지
    async function sendToServer(userMessage, loadingElement) {
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            });
            
            // 로딩 표시 제거
            loadingElement.remove();
            
            if (!response.ok) {
                throw new Error(`서버 오류: ${response.status}`);
            }
            
            const data = await response.json();
            
            // 서버 응답 추가
            if (data && data.response) {
                addMessage(data.response, 'bot');
            } else {
                throw new Error('서버 응답 형식이 올바르지 않습니다.');
            }
        } catch (error) {
            console.error('Error:', error);
            loadingElement.remove();
            
            // 오류 메시지 표시
            addMessage('죄송합니다. 서버 연결에 문제가 발생했습니다: ' + error.message, 'bot', true);
        } finally {
            // 입력 상태 업데이트
            isWaitingForResponse = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    // 새 채팅 버튼
    newChatBtn.addEventListener('click', function() {
        // 메시지 초기화 (첫 번째 인사말만 남김)
        while (messagesContainer.children.length > 1) {
            messagesContainer.removeChild(messagesContainer.lastChild);
        }
        
        // 인트로 화면 표시, 채팅 화면 숨김
        landingPage.classList.remove('hidden');
        chatInterface.classList.add('hidden');
    });

    // 예시 질문 클릭 시 입력 필드에 적용
    window.setExampleQuestion = function(question) {
        messageInput.value = question;
        messageInput.focus();
        // 텍스트 영역 높이 자동 조절
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight > 150 ? 150 : messageInput.scrollHeight) + 'px';
    };
});
