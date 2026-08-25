document.addEventListener('DOMContentLoaded', () => {

    // элементы страницы ====================================================================
    const newChatBtn  = document.getElementById('new-chat-btn');
    const mobileDialogsToggle = document.getElementById('mobile-dialogs-toggle');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');

    // вкладки
    const tabBtns     = document.querySelectorAll('.tab');
    const tabChat     = document.getElementById('tab-chat');
    const tabUpload   = document.getElementById('tab-upload');

    // чат
    const chatArea    = document.getElementById('chat-area');
    const chatForm    = document.getElementById('chat-form');
    const msgInput    = document.getElementById('message-input');
    const sendBtn     = document.getElementById('send-btn');

    // загрузка
    const dropZone        = document.getElementById('drop-zone');
    const fileInput       = document.getElementById('file-input');
    const filePickerBtn   = document.getElementById('file-picker-btn');
    const pipelineSection = document.getElementById('pipeline-section');
    const pipelineFilename= document.getElementById('pipeline-filename');
    const progressBar     = document.getElementById('progress-bar');
    const statusText      = document.getElementById('status-text');
    const stageDetail     = document.getElementById('stage-detail');
    const checklistCard    = document.getElementById('checklist-card');
    const checklistContent= document.getElementById('checklist-content');
    const summaryCard     = document.getElementById('summary-card');
    const summaryContent  = document.getElementById('summary-content');
    const docsList        = document.getElementById('docs-list');
    const docsEmpty       = document.getElementById('docs-empty');

    // дополнительные элементы для нескольких чатов
    const chatList       = document.getElementById('chat-list');
    // модальное окно
    const docModal        = document.getElementById('doc-modal');
    const modalClose      = document.getElementById('modal-close');
    const modalBody       = document.getElementById('modal-body');
    const modalTitle      = document.getElementById('modal-title');

    const setMobileSidebarOpen = (open) => {
        const canOpen = window.matchMedia('(max-width: 720px)').matches
            && !tabChat.classList.contains('hidden');
        const shouldOpen = open && canOpen;

        document.body.classList.toggle('mobile-sidebar-open', shouldOpen);
        sidebarBackdrop.classList.toggle('hidden', !shouldOpen);
        mobileDialogsToggle.setAttribute('aria-expanded', String(shouldOpen));
        mobileDialogsToggle.setAttribute(
            'aria-label',
            shouldOpen ? 'Закрыть диалоги' : 'Открыть диалоги'
        );
    };

    const closeMobileSidebar = () => setMobileSidebarOpen(false);

    mobileDialogsToggle.addEventListener('click', () => {
        setMobileSidebarOpen(!document.body.classList.contains('mobile-sidebar-open'));
    });
    sidebarBackdrop.addEventListener('click', closeMobileSidebar);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeMobileSidebar();
    });
    window.addEventListener('resize', () => {
        if (!window.matchMedia('(max-width: 720px)').matches) closeMobileSidebar();
    });

    if (modalClose) {
        modalClose.addEventListener('click', () => {
            docModal.classList.add('hidden');
        });
    }

    if (docModal) {
        docModal.addEventListener('click', (e) => {
            if (e.target === docModal) {
                docModal.classList.add('hidden');
            }
        });
    }

    // управление чатами ====================================================================
    let currentChatId = null;
    let chatsData = [];

    async function loadChats() {
        try {
            const res = await fetch('/api/chats');
            if (res.ok) {
                const data = await res.json();
                chatsData = data.chats;
                renderChatList();
                if (chatsData.length > 0 && !currentChatId) {
                    selectChat(chatsData[0].chat_id);
                } else if (!currentChatId) {
                    resetChatArea();
                }
            }
        } catch (e) {
            console.error('не удалось загрузить чаты:', e);
        }
    }

    function renderChatList() {
        chatList.innerHTML = '';
        if (chatsData.length === 0) {
            chatList.innerHTML = '<div class="sidebar__empty">Нет диалогов</div>';
            return;
        }

        chatsData.forEach(chat => {
            const item = document.createElement('div');
            item.className = `dialog ${chat.chat_id === currentChatId ? 'dialog--active' : ''}`;
            
            const title = document.createElement('div');
            title.className = 'dialog__title';
            title.textContent = chat.title || 'Новый чат';
            item.appendChild(title);

            const delBtn = document.createElement('button');
            delBtn.className = 'dialog__delete';
            delBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>';
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChat(chat.chat_id);
            });
            item.appendChild(delBtn);

            item.addEventListener('click', () => selectChat(chat.chat_id));
            chatList.appendChild(item);
        });
    }

    async function createNewChat() {
        try {
            const res = await fetch('/api/chats', { method: 'POST' });
            if (res.ok) {
                const chat = await res.json();
                chatsData.unshift(chat); // добавить наверх
                selectChat(chat.chat_id);
                renderChatList();
            }
        } catch (e) {
            console.error('не удалось создать чат:', e);
        }
    }

    async function deleteChat(id) {
        if (!confirm('Удалить этот диалог?')) return;
        try {
            const res = await fetch(`/api/chats/${id}`, { method: 'DELETE' });
            if (res.ok) {
                chatsData = chatsData.filter(c => c.chat_id !== id);
                if (currentChatId === id) {
                    currentChatId = null;
                    if (chatsData.length > 0) selectChat(chatsData[0].chat_id);
                    else resetChatArea();
                }
                renderChatList();
            }
        } catch (e) {
            console.error('не удалось удалить чат:', e);
        }
    }

    async function selectChat(id) {
        currentChatId = id;
        closeMobileSidebar();
        renderChatList(); // обновить активный класс
        chatArea.innerHTML = '<div class="dots dots--center"><span class="dots__item"></span><span class="dots__item"></span><span class="dots__item"></span></div>';
        
        try {
            const res = await fetch(`/api/chats/${id}/messages`);
            if (res.ok) {
                const data = await res.json();
                chatArea.innerHTML = '';
                if (data.messages.length === 0) {
                    resetChatArea();
                } else {
                    data.messages.forEach(msg => {
                        const rendered = renderMarkdown(msg.content);
                        addMessage(
                            msg.role,
                            rendered,
                            msg.model || null,
                            msg.sources || []
                        );
                    });
                }
            }
        } catch (e) {
            console.error('не удалось загрузить сообщения:', e);
            resetChatArea();
        }
    }

    function resetChatArea() {
        chatArea.innerHTML = `
            <div class="welcome" id="welcome">
                <h1 class="welcome__title">Добро пожаловать!</h1>
                <p class="welcome__text">Задавайте вопросы по загруженным лекционным материалам. Ответы основаны на содержании ваших лекций.</p>
            </div>
        `;
    }

    // загрузить список чатов
    loadChats();

    // кнопки нового чата ====================================================================
    newChatBtn.addEventListener('click', () => createNewChat());

    // переключение вкладок ====================================================================
    const tabs = { chat: tabChat, upload: tabUpload };

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('tab--active'));
            btn.classList.add('tab--active');
            Object.values(tabs).forEach(t => t.classList.add('hidden'));
            tabs[target].classList.remove('hidden');
            mobileDialogsToggle.classList.toggle('hidden', target !== 'chat');
            if (target !== 'chat') closeMobileSidebar();

            if (target === 'upload') refreshDocuments();
        });
    });

    // менять высоту поля по тексту ====================================================================
    msgInput.addEventListener('input', () => {
        msgInput.style.height = 'auto';
        msgInput.style.height = msgInput.scrollHeight + 'px';
        sendBtn.disabled = msgInput.value.trim().length === 0;
    });

    msgInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) chatForm.dispatchEvent(new Event('submit'));
        }
    });
    sendBtn.disabled = true;

    // функции чата ====================================================================
    const scrollToBottom = () => { chatArea.scrollTop = chatArea.scrollHeight; };

    const escapeHtml = (text) => {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    };

    const renderMarkdown = (text) => DOMPurify.sanitize(marked.parse(text ?? ""));


    const formatKindLabel = (kind) => {
        if (kind === 'summary') return 'Конспект';
        if (kind === 'transcript') return 'Транскрипт';
        return 'Материал';
    };

    const formatSourceMeta = (source) => {
        const parts = [formatKindLabel(source.content_kind)];
        if (source.chunk_index !== undefined && source.chunk_index !== null) {
            parts.push(`фрагмент #${source.chunk_index}`);
        }
        if (Number.isFinite(source.rerank_score)) {
            parts.push(`релевантность ${source.rerank_score.toFixed(2)}`);
        }
        return parts.join(' · ');
    };

    const sortSourcesByRelevance = (sources) => {
        return [...sources].sort((a, b) => {
            const aScore = Number.isFinite(a.rerank_score) ? a.rerank_score : null;
            const bScore = Number.isFinite(b.rerank_score) ? b.rerank_score : null;

            if (aScore === null && bScore === null) return 0;
            if (aScore === null) return 1;
            if (bScore === null) return -1;
            return bScore - aScore;
        });
    };

    const renderMaterialSection = (title, content) => {
        if (!content || !content.trim()) return '';
        return `
            <section class="material">
                <h2 class="material__title">${escapeHtml(title)}</h2>
                ${renderMarkdown(content)}
            </section>
        `;
    };

    const applyMathRendering = (element) => {
        if (window.renderMathInElement && element) {
            window.renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false
            });
        }
    };

    const addMessage = (role, html, modelName = null, sources = []) => {
        const welcome = document.getElementById('welcome');
        if (welcome) welcome.remove();

        const msg = document.createElement('div');
        msg.className = `message message--${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message__bubble prose';
        bubble.innerHTML = html;
        msg.appendChild(bubble);

        if (role === 'assistant' && modelName) {
            const badge = document.createElement('span');
            badge.className = 'message__model';
            badge.textContent = `Ответ от: ${modelName}`;
            msg.appendChild(badge);
        }

        if (role === 'assistant' && sources.length > 0) {
            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'message__sources';
            toggle.textContent = `Источники: ${sources.length}`;

            const list = document.createElement('div');
            list.className = 'message__sources-list hidden';

            sortSourcesByRelevance(sources).forEach(s => {
                const item = document.createElement('div');
                item.className = 'source';

                const header = document.createElement('div');
                header.className = 'source__name';
                header.textContent = s.filename || 'Без названия';

                const meta = document.createElement('div');
                meta.className = 'source__meta';
                meta.textContent = formatSourceMeta(s);

                const preview = document.createElement('div');
                preview.className = 'source__preview';
                preview.textContent = s.text_preview || '';

                item.appendChild(header);
                item.appendChild(meta);
                item.appendChild(preview);

                list.appendChild(item);
            });

            toggle.addEventListener('click', () => list.classList.toggle('hidden'));
            msg.appendChild(toggle);
            msg.appendChild(list);
        }

        chatArea.appendChild(msg);
        applyMathRendering(msg);
        scrollToBottom();
        return msg;
    };

    const addLoading = () => {
        const welcome = document.getElementById('welcome');
        if (welcome) welcome.remove();
        const msg = document.createElement('div');
        msg.className = 'message assistant';
        msg.id = 'loading-msg';
        const bubble = document.createElement('div');
        bubble.className = 'message__bubble prose';
        bubble.innerHTML = '<div class="dots"><span class="dots__item"></span><span class="dots__item"></span><span class="dots__item"></span></div>';
        msg.appendChild(bubble);
        chatArea.appendChild(msg);
        scrollToBottom();
        return msg;
    };

    const removeLoading = () => {
        const el = document.getElementById('loading-msg');
        if (el) el.remove();
    };

    // отправка сообщения ====================================================================
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = msgInput.value.trim();
        if (!text) return;

        // если чат не выбран, создать его
        if (!currentChatId) {
            try {
                const crc = await fetch('/api/chats', { method: 'POST' });
                if (crc.ok) {
                    const chat = await crc.json();
                    chatsData.unshift(chat);
                    currentChatId = chat.chat_id;
                    renderChatList();
                } else {
                    throw new Error("Failed to create chat");
                }
            } catch (err) {
                console.error(err);
                return;
            }
        }

        msgInput.value = '';
        msgInput.style.height = 'auto';
        sendBtn.disabled = true;

        addMessage('user', escapeHtml(text));
        addLoading();

        try {
            const payload = {
                message: text,
                chat_id: currentChatId
            };

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            removeLoading();

            if (res.ok) {
                const data = await res.json();
                const rendered = renderMarkdown(data.answer || '');
                addMessage('assistant', rendered, data.model || null, data.sources || []);
                
                // обновить заголовок, если он создан
                if (data.title) {
                    const idx = chatsData.findIndex(c => c.chat_id === currentChatId);
                    if (idx !== -1 && chatsData[idx].title !== data.title) {
                        chatsData[idx].title = data.title;
                        renderChatList();
                    }
                }
            } else {
                let detail = 'Произошла ошибка.';
                try { const err = await res.json(); detail = err.detail || detail; } catch (_) {}
                addMessage('assistant', `<span class="error-text">${escapeHtml(detail)}</span>`);
            }
        } catch (err) {
            removeLoading();
            addMessage('assistant', '<span class="error-text">Ошибка сети. Убедитесь, что сервер запущен.</span>');
            console.error(err);
        }
    });

    // выбор файла ====================================================================
    filePickerBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dropzone--active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dropzone--active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dropzone--active');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });

    // нажатие на область открывает выбор файла
    dropZone.addEventListener('click', (e) => {
        if (e.target !== filePickerBtn) fileInput.click();
    });

    // загрузка файла ====================================================================
    async function uploadFile(file) {
        // показать ход обработки
        pipelineSection.classList.remove('hidden');
        pipelineFilename.textContent = file.name;
        progressBar.style.setProperty('--pipeline-progress', '0%');
        progressBar.className = 'pipeline__bar';
        statusText.textContent = 'Загрузка файла…';
        statusText.className = 'pipeline__status';
        stageDetail.classList.add('hidden');
        stageDetail.textContent = '';
        checklistCard.classList.add('hidden');
        summaryCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });

            if (!res.ok) {
                let detail = 'Ошибка загрузки';
                try { const err = await res.json(); detail = err.detail || detail; } catch (_) {}
                statusText.textContent = detail;
                statusText.className = 'pipeline__status pipeline__status--error';
                progressBar.className = 'pipeline__bar pipeline__bar--error';
                progressBar.style.setProperty('--pipeline-progress', '100%');
                return;
            }

            const { job_id } = await res.json();
            pollJobStatus(job_id);

        } catch (err) {
            statusText.textContent = 'Ошибка сети при загрузке';
            statusText.className = 'pipeline__status pipeline__status--error';
            console.error(err);
        }
    }

    async function pollJobStatus(jobId) {
        const poll = async () => {
            try {
                const res = await fetch(`/api/jobs/${jobId}`);
                if (!res.ok) return;

                const job = await res.json();

                progressBar.style.setProperty('--pipeline-progress', `${job.progress}%`);
                statusText.textContent = job.stage;

                if (job.transcription && job.transcription.total) {
                    stageDetail.classList.remove('hidden');
                    stageDetail.textContent = `${job.transcription.current}/${job.transcription.total} фрагментов · ${job.transcription.percent}%`;
                } else {
                    stageDetail.classList.add('hidden');
                    stageDetail.textContent = '';
                }

                // показать чек-лист и конспект сразу после готовности
                if (job.checklist && checklistCard.classList.contains('hidden')) {
                    checklistCard.classList.remove('hidden');
                    checklistContent.innerHTML = renderMarkdown(job.checklist);
                    applyMathRendering(checklistContent);
                }
                if (job.summary && summaryCard.classList.contains('hidden')) {
                    summaryCard.classList.remove('hidden');
                    summaryContent.innerHTML = renderMarkdown(job.summary);
                    applyMathRendering(summaryContent);
                }

                if (job.status === 'done') {
                    progressBar.className = 'pipeline__bar pipeline__bar--done';
                    statusText.className = 'pipeline__status pipeline__status--done';
                    statusText.textContent = `Готово. Проиндексировано ${job.chunk_count} фрагментов.`;

                    refreshDocuments();
                    return; // остановить проверку
                }

                if (job.status === 'error') {
                    progressBar.className = 'pipeline__bar pipeline__bar--error';
                    progressBar.style.setProperty('--pipeline-progress', '100%');
                    statusText.className = 'pipeline__status pipeline__status--error';
                    statusText.textContent = `Ошибка: ${job.error}`;
                    return; // остановить проверку
                }

                // обработка еще идет, проверить снова
                setTimeout(poll, 2000);

            } catch (err) {
                console.error('ошибка проверки состояния:', err);
                setTimeout(poll, 5000);
            }
        };
        setTimeout(poll, 1500); // первая пауза
    }

    // список документов ====================================================================
    async function refreshDocuments() {
        try {
            const res = await fetch('/api/documents');
            if (!res.ok) return;

            const { documents } = await res.json();

            if (documents.length === 0) {
                docsList.innerHTML = '<p class="note" id="docs-empty">Нет загруженных материалов</p>';
                return;
            }

            docsList.innerHTML = '';
            documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc';
                item.innerHTML = `
                    <div class="doc__info">
                        <span class="doc__name">${escapeHtml(doc.filename)}</span>
                        <span class="doc__meta">${doc.source_type} · ${doc.chunk_count} фрагментов · конспект</span>
                    </div>
                `;

                const actions = document.createElement('div');
                actions.className = 'doc__actions';

                const viewBtn = document.createElement('button');
                viewBtn.className = 'doc__btn doc__btn--view';
                viewBtn.textContent = 'Открыть материал';
                viewBtn.addEventListener('click', () => viewDocument(doc.doc_id));

                const delBtn = document.createElement('button');
                delBtn.className = 'doc__btn doc__btn--delete';
                delBtn.textContent = 'Удалить';
                delBtn.addEventListener('click', () => deleteDocument(doc.doc_id, item));

                actions.appendChild(viewBtn);
                actions.appendChild(delBtn);
                item.appendChild(actions);
                docsList.appendChild(item);
            });

        } catch (err) {
            console.error('не удалось загрузить документы:', err);
        }
    }

    async function viewDocument(docId) {
        docModal.classList.remove('hidden');
        modalTitle.textContent = 'Материалы лекции';
        modalBody.innerHTML = '<div class="dots dots--center"><span class="dots__item"></span><span class="dots__item"></span><span class="dots__item"></span></div>';

        try {
            const res = await fetch(`/api/documents/${docId}`);
            if (res.ok) {
                const data = await res.json();
                const doc = data.document;
                if (doc) {
                    modalTitle.textContent = doc.filename || 'Материалы лекции';
                    const sections = renderMaterialSection('Конспект', doc.display_summary);
                    modalBody.innerHTML = sections || '<p class="note">Материалы пока пусты.</p>';
                    applyMathRendering(modalBody);
                } else {
                    modalBody.innerHTML = '<p class="note">Материалы не найдены.</p>';
                }
            } else {
                modalBody.innerHTML = '<p class="error-text">Ошибка загрузки материалов.</p>';
            }
        } catch (e) {
            console.error('не удалось загрузить документ:', e);
            modalBody.innerHTML = '<p class="error-text">Ошибка сети.</p>';
        }
    }

    async function deleteDocument(docId, element) {
        if (!confirm('Удалить этот документ из базы знаний?')) return;

        try {
            const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
            if (res.ok) {
                element.remove();
                // проверить, остались ли документы
                if (docsList.children.length === 0) {
                    docsList.innerHTML = '<p class="note">Нет загруженных материалов</p>';
                }
            }
        } catch (err) {
            console.error('не удалось удалить документ:', err);
        }
    }

    // первая загрузка списка документов
    refreshDocuments();
});
