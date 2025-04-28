document.addEventListener('DOMContentLoaded', function() {
    // 파일 정보를 불러오는 함수
    async function fetchFileInfo() {
        try {
            const response = await fetch('/api/files');
            if (!response.ok) {
                throw new Error('파일 정보를 불러오는 데 실패했습니다.');
            }
            return await response.json();
        } catch (error) {
            console.error('에러:', error);
            showErrorMessage('파일 정보를 불러오는 데 실패했습니다.');
            return null;
        }
    }

    // 다운로드 링크 설정 함수
    function setupDownloadLinks(filesData) {
        // 메인 파일(통합 문서) 다운로드 설정
        const mainDownloadBtn = document.getElementById('download-full');
        if (mainDownloadBtn && filesData.full) {
            mainDownloadBtn.href = filesData.full.path;
            mainDownloadBtn.setAttribute('download', filesData.full.title);
            
            // 썸네일 이미지가 있으면 설정
            const thumbnail = document.getElementById('pdf-thumbnail');
            if (thumbnail && filesData.full.thumbnail) {
                thumbnail.src = filesData.full.thumbnail;
                thumbnail.alt = `${filesData.full.title} 썸네일`;
            }
        }

        // 개별 파일 다운로드 버튼 설정
        const downloadButtons = document.querySelectorAll('.download-button');
        downloadButtons.forEach(button => {
            const fileId = button.getAttribute('data-file');
            if (filesData[fileId]) {
                button.href = filesData[fileId].path;
                button.setAttribute('download', filesData[fileId].title);
            }
        });
    }

    // 다운로드 이벤트 처리
    function handleDownload(event) {
        const downloadBtn = event.currentTarget;
        const fileId = downloadBtn.getAttribute('data-file');
        const filePath = downloadBtn.getAttribute('href');
        
        if (!filePath || filePath === '#') {
            event.preventDefault();
            showErrorMessage('다운로드 링크가 설정되지 않았습니다.');
            return;
        }

        // 다운로드 통계 등을 수집할 수 있습니다.
        console.log(`파일 다운로드: ${fileId || 'full'}`);

        // 기본 브라우저 다운로드 동작 허용
    }

    // 오류 메시지 표시
    function showErrorMessage(message) {
        // 오류 메시지를 표시할 요소 생성
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        const container = document.querySelector('.container');
        if (container) {
            container.prepend(errorDiv);
            
            // 5초 후 오류 메시지 제거
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        } else {
            alert(message);
        }
    }

    // 파일 다운로드 이벤트 리스너 설정
    function setupEventListeners() {
        // 메인 파일 다운로드 버튼 이벤트
        const mainDownloadBtn = document.getElementById('download-full');
        if (mainDownloadBtn) {
            mainDownloadBtn.addEventListener('click', handleDownload);
        }

        // 개별 파일 다운로드 버튼 이벤트
        const downloadButtons = document.querySelectorAll('.download-button');
        downloadButtons.forEach(button => {
            button.addEventListener('click', handleDownload);
        });
    }

    // 로딩 상태 표시 함수
    function showLoading(show = true) {
        // 이미 로딩 요소가 있는지 확인
        let loadingEl = document.querySelector('.loading-indicator');
        
        if (show) {
            if (!loadingEl) {
                loadingEl = document.createElement('div');
                loadingEl.className = 'loading-indicator';
                loadingEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 로딩 중...';
                
                document.body.appendChild(loadingEl);
            }
        } else if (loadingEl) {
            loadingEl.remove();
        }
    }

    // 초기화 함수
    async function init() {
        showLoading(true);
        
        try {
            const filesData = await fetchFileInfo();
            if (filesData) {
                setupDownloadLinks(filesData);
                setupEventListeners();
            }
        } catch (error) {
            console.error('초기화 중 오류:', error);
            showErrorMessage('페이지 초기화 중 오류가 발생했습니다.');
        } finally {
            showLoading(false);
        }
    }

    // 앱 초기화
    init();

    // 썸네일 클릭 시 다운로드 버튼 클릭 이벤트 트리거
    const thumbnail = document.getElementById('pdf-thumbnail');
    if (thumbnail) {
        thumbnail.addEventListener('click', function() {
            const downloadBtn = document.getElementById('download-full');
            if (downloadBtn) {
                downloadBtn.click();
            }
        });
    }
});