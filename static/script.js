document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    let isWaitingForResponse = false;

    // API 기본 URL
    const API_URL = '/api/chat';

    // 텍스트 영역 자동 높이 조절
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight > 150 ? 150 : messageInput.scrollHeight) + 'px';
    });

    // 엔터 키로 메시지 전송 (Shift+Enter는 줄바꿈)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 전송 버튼 클릭으로 메시지 전송
    sendButton.addEventListener('click', sendMessage);

    // 메시지 전송 함수
    function sendMessage() {
        const message = messageInput.value.trim();
        
        if (message && !isWaitingForResponse) {
            // 사용자 메시지 추가
            addMessage(message, 'user');
            
            // 입력 필드 초기화
            messageInput.value = '';
            messageInput.style.height = '50px';
            
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
            
            // 서버와 통신
            sendToServer(message, loadingElement);
        }
    }

    // 메시지 추가 함수
    function addMessage(text, sender, isError = false) {
        const messageElement = document.createElement('div');
        
        if (isError) {
            messageElement.className = 'message error-message';
        } else {
            messageElement.className = `message ${sender}-message`;
        }
        
        messageElement.textContent = text;
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }

    // 맨 아래로 스크롤
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // 서버로 메시지 전송 및 응답 처리
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
});
