document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    let isWaitingForResponse = false;

    // API 기본 URL - 서버의 URL에 맞게 수정
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

    // 세션 ID 생성 함수
    function generateSessionId() {
        return 'session_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }

    // 메시지 제거 함수
    function removeMessage(messageId) {
        const messageElement = document.getElementById(messageId);
        if (messageElement) {
            messageElement.remove();
        }
    }

    // 메시지 추가 함수
    function addMessage(text, sender, isError = false) {
        const messageElement = document.createElement('div');
        
        // 고유 ID 생성
        const messageId = 'msg_' + Math.random().toString(36).substring(2, 9);
        messageElement.id = messageId;
        
        if (isError) {
            messageElement.className = 'message error-message';
        } else {
            messageElement.className = `message ${sender}-message`;
        }
        
        // 텍스트에 줄바꿈이 있으면 HTML에서도 줄바꿈 처리
        messageElement.innerHTML = text.replace(/\n/g, '<br>');
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
        
        // 메시지 ID 반환
        return messageId;
    }

    // 메시지 전송 함수
    async function sendMessage() {
        const userMessage = document.getElementById('message-input').value.trim();
        if (!userMessage) return;
        
        // 사용자 메시지 UI에 추가
        addMessage(userMessage, 'user');
        document.getElementById('message-input').value = '';
        
        // 응답 대기 중 표시
        const waitingId = addMessage('...', 'assistant');
        
        try {
            // 세션 ID는 브라우저에 저장
            const sessionId = localStorage.getItem('session_id') || generateSessionId();
            localStorage.setItem('session_id', sessionId);
            
            // 채팅 API 호출 (스트리밍 방식)
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    session_id: sessionId,
                    stream_mode: true  // 스트리밍 모드 활성화
                })
            });
            
            // 응답 형식 확인
            const contentType = response.headers.get('content-type');
            
            // 스트리밍 응답 처리
            if (contentType && contentType.includes('application/x-ndjson')) {
                // 대기 메시지 제거
                removeMessage(waitingId);
                
                // 스트리밍 응답을 위한 새 메시지 요소 생성
                const streamId = addMessage('', 'assistant');
                const messageElement = document.getElementById(streamId);
                
                // 스트림 리더 생성
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let streamedText = '';
                
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    // 청크 디코딩
                    const chunk = decoder.decode(value, { stream: true });
                    
                    // 각 라인 처리
                    chunk.split('\n').filter(line => line.trim()).forEach(line => {
                        try {
                            const data = JSON.parse(line);
                            
                            if (data.type === 'token') {
                                // 토큰 추가
                                streamedText += data.content;
                                messageElement.innerHTML = formatMessage(streamedText);
                            } else if (data.type === 'end') {
                                // 스트리밍 완료, 최종 메시지로 업데이트
                                messageElement.innerHTML = formatMessage(data.content);
                            } else if (data.type === 'error') {
                                // 오류 메시지 표시
                                messageElement.innerHTML = formatMessage(data.content);
                                messageElement.classList.add('error-message');
                            }
                        } catch (e) {
                            console.error('Invalid JSON chunk:', line, e);
                        }
                    });
                }
            } else {
                // 기존 방식의 응답 처리 (스트리밍 아닌 경우)
                const data = await response.json();
                removeMessage(waitingId);
                addMessage(data.response, 'assistant');
            }
        } catch (error) {
            console.error('Error:', error);
            removeMessage(waitingId);
            addMessage('죄송합니다. 요청 처리 중 오류가 발생했습니다.', 'assistant');
        }
    }


    function formatMessage(text) {
        // HTML 이스케이핑 처리
        let formatted = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // 줄바꿈 처리
        formatted = formatted.replace(/\n/g, '<br>');
        
        // 추가적인 포맷팅 규칙 적용 가능
        
        return formatted;
    }

    // 메시지 추가 함수
    function addMessage(text, sender, isError = false) {
        const messageElement = document.createElement('div');
        
        if (isError) {
            messageElement.className = 'message error-message';
        } else {
            messageElement.className = `message ${sender}-message`;
        }
        
        // 텍스트에 줄바꿈이 있으면 HTML에서도 줄바꿈 처리
        messageElement.innerHTML = text.replace(/\n/g, '<br>');
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }

    // 맨 아래로 스크롤
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // 서버로 메시지 전송 및 응답 처리
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
            console.log("서버 응답:", data);  // 응답 로깅
            
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