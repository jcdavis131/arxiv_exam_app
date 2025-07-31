// arXiv Exam Generator - Client-side functionality with Search
class ExamApp {
    constructor() {
        this.elements = {
            arxivInput: document.getElementById('arxivIdInput'),
            generateBtn: document.getElementById('generateExamBtn'),
            examContainer: document.getElementById('examContainer'),
            errorContainer: document.getElementById('errorContainer'),
            loadingIndicator: document.getElementById('loadingIndicator'),
            teacherModeToggle: document.getElementById('teacherModeToggle'),
            llmProvider: document.getElementById('llmProvider'),
            llmModel: document.getElementById('llmModel'),
            apiKey: document.getElementById('apiKey'),
            saveConfig: document.getElementById('saveConfig'),
            configStatus: document.getElementById('configStatus'),
            downloadExamBtn: document.getElementById('downloadExamBtn'),
            downloadSinglePdfBtn: document.getElementById('downloadSinglePdfBtn'),
            
            // Search elements
            modeButtons: document.querySelectorAll('.mode-btn'),
            singlePaperMode: document.getElementById('singlePaperMode'),
            searchMode: document.getElementById('searchMode'),
            singlePaperInput: document.getElementById('singlePaperInput'),
            searchInput: document.getElementById('searchInput'),
            searchTabs: document.querySelectorAll('.search-tab'),
            searchPanels: document.querySelectorAll('.search-panel'),
            performSearchBtn: document.getElementById('performSearchBtn'),
            searchResults: document.getElementById('searchResults'),
            resultsList: document.getElementById('resultsList'),
            generateFromSearchBtn: document.getElementById('generateFromSearchBtn'),
            
            // Search inputs
            generalQuery: document.getElementById('generalQuery'),
            generalSort: document.getElementById('generalSort'),
            authorQuery: document.getElementById('authorQuery'),
            authorSort: document.getElementById('authorSort'),
            categorySelect: document.getElementById('categorySelect'),
            categorySort: document.getElementById('categorySort'),
            dateFrom: document.getElementById('dateFrom'),
            dateTo: document.getElementById('dateTo'),
            maxPapers: document.getElementById('maxPapers'),
            mcQuestions: document.getElementById('mcQuestions'),
            oeQuestions: document.getElementById('oeQuestions'),
            examGenerationControls: document.querySelector('.exam-generation-controls'),
        };
        
        this.state = {
            currentQuestions: [],
            currentExamData: null,
            examSubmitted: false,
            teacherMode: false,
            llmConfig: this.loadConfig(),
            currentMode: 'single',
            currentSearchType: 'general',
            searchResults: [],
            selectedPapers: [],
            currentPage: 1
        };
        
        this.modelOptions = {
            openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
            groq: ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
            huggingface: ['microsoft/DialoGPT-large', 'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-small']
        };
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeConfig();
        this.initializeSearch();
        this.elements.arxivInput.focus();
    }

    loadConfig() {
        const saved = localStorage.getItem('llmConfig');
        return saved ? JSON.parse(saved) : {
            provider: 'openai',
            model: 'gpt-4o-mini',
            apiKey: ''
        };
    }

    saveConfig() {
        const config = {
            provider: this.elements.llmProvider.value,
            model: this.elements.llmModel.value,
            apiKey: this.elements.apiKey.value
        };
        
        localStorage.setItem('llmConfig', JSON.stringify(config));
        this.state.llmConfig = config;
        
        this.showConfigStatus('Configuration saved successfully!', 'success');
        setTimeout(() => this.clearConfigStatus(), 3000);
    }

    initializeConfig() {
        // Set initial values
        this.elements.llmProvider.value = this.state.llmConfig.provider;
        this.elements.apiKey.value = this.state.llmConfig.apiKey;
        
        // Update model options
        this.updateModelOptions();
        this.elements.llmModel.value = this.state.llmConfig.model;
    }

    updateModelOptions() {
        const provider = this.elements.llmProvider.value;
        const models = this.modelOptions[provider] || [];
        
        this.elements.llmModel.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            this.elements.llmModel.appendChild(option);
        });
    }

    showConfigStatus(message, type = 'info') {
        this.elements.configStatus.textContent = message;
        this.elements.configStatus.className = `config-status ${type}`;
    }

    clearConfigStatus() {
        this.elements.configStatus.textContent = '';
        this.elements.configStatus.className = 'config-status';
    }

    toggleTeacherMode() {
        this.state.teacherMode = !this.state.teacherMode;
        this.elements.teacherModeToggle.classList.toggle('active', this.state.teacherMode);
        
        // Update main container class for styling
        const container = document.querySelector('.container');
        container.classList.toggle('teacher-mode-active', this.state.teacherMode);
        
        // If exam is already loaded, update display
        if (this.state.currentQuestions.length > 0) {
            this.updateTeacherModeDisplay();
        }
    }

    updateTeacherModeDisplay() {
        if (this.state.teacherMode) {
            this.showAllAnswers();
        } else {
            this.hideAllAnswers();
        }
    }

    showAllAnswers() {
        this.state.currentQuestions.forEach((question, index) => {
            const questionDiv = document.querySelector(`[data-question-index="${index}"]`);
            if (questionDiv) {
                // Remove existing teacher answers
                const existingTeacherAnswers = questionDiv.querySelector('.teacher-answers');
                if (existingTeacherAnswers) {
                    existingTeacherAnswers.remove();
                }
                
                // Add teacher answers
                const teacherAnswers = this.createTeacherAnswers(question);
                questionDiv.appendChild(teacherAnswers);
            }
        });
    }

    hideAllAnswers() {
        const teacherAnswers = document.querySelectorAll('.teacher-answers');
        teacherAnswers.forEach(el => el.remove());
    }

    bindEvents() {
        this.elements.generateBtn.addEventListener('click', () => this.generateExam());
        this.elements.arxivInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.generateExam();
        });
        this.elements.arxivInput.addEventListener('input', () => this.hideError());
        this.elements.teacherModeToggle.addEventListener('click', () => this.toggleTeacherMode());
        
        // PDF Download for single paper
        this.elements.downloadSinglePdfBtn.addEventListener('click', () => {
            const arxivId = this.elements.arxivInput.value.trim();
            if (arxivId && this.validateArxivId(arxivId)) {
                this.downloadPDF(arxivId);
            } else {
                this.showError('Please enter a valid arXiv paper ID.');
            }
        });
        
        // LLM Configuration events
        this.elements.llmProvider.addEventListener('change', () => this.updateModelOptions());
        this.elements.saveConfig.addEventListener('click', () => this.saveConfig());
        this.elements.apiKey.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.saveConfig();
        });

        // Download Exam events
        this.elements.downloadExamBtn.addEventListener('click', () => {
            if (this.state.currentQuestions.length > 0 && this.state.currentExamData) {
                this.downloadEnhancedExam();
            } else {
                alert('No exam data to download. Please generate an exam first.');
            }
        });
    }

    validateArxivId(id) {
        const patterns = [
            /^\d{4}\.\d{4,5}$/,           // 2406.01234
            /^[a-zA-Z-]+\/\d{7}$/,       // hep-th/0401001
            /^[a-zA-Z-]+\.\w{2}\/\d{7}$/ // math.AG/0601001
        ];
        return patterns.some(pattern => pattern.test(id));
    }

    showError(message) {
        this.elements.errorContainer.innerHTML = `<strong>Error:</strong> ${message}`;
        this.elements.errorContainer.style.display = 'block';
    }

    hideError() {
        this.elements.errorContainer.innerHTML = '';
        this.elements.errorContainer.style.display = 'none';
    }

    showLoading() {
        this.elements.loadingIndicator.style.display = 'block';
        this.elements.generateBtn.disabled = true;
        this.elements.generateBtn.textContent = 'Processing...';
        this.resetProgressBar();
    }

    hideLoading() {
        this.elements.loadingIndicator.style.display = 'none';
        this.elements.generateBtn.disabled = false;
        this.elements.generateBtn.textContent = 'Generate Exam';
    }
    
    updateProgressSteps(isMultiPaper = false, mcQuestions = 7, oeQuestions = 3) {
        const steps = [
            {
                single: "Fetching paper content",
                multi: "Processing selected papers"
            },
            {
                single: "Analyzing document structure", 
                multi: "Downloading and analyzing papers"
            },
            {
                single: "Processing with AI",
                multi: "Combining content from multiple papers"
            },
            {
                single: `Generating ${mcQuestions} multiple-choice questions`,
                multi: `Generating ${mcQuestions} multiple-choice questions`
            },
            {
                single: `Generating ${oeQuestions} open-ended questions`,
                multi: `Generating ${oeQuestions} open-ended questions`
            },
            {
                single: "Finalizing and shuffling answers",
                multi: "Finalizing and shuffling answers"
            },
            {
                single: "Naming exam",
                multi: "Naming exam"
            }
        ];
        
        steps.forEach((step, index) => {
            const stepElement = document.getElementById(`step${index + 1}`);
            if (stepElement) {
                const textElement = stepElement.querySelector('.step-text');
                if (textElement) {
                    textElement.textContent = isMultiPaper ? step.multi : step.single;
                }
            }
        });
    }

    resetProgressBar() {
        // Reset all progress elements
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressPercentage').textContent = '0%';
        
        // Reset all steps
        for (let i = 1; i <= 7; i++) {
            const step = document.getElementById(`step${i}`);
            step.classList.remove('active', 'completed');
        }
    }

    updateProgress(step, percentage) {
        // Update percentage and fill
        document.getElementById('progressFill').style.width = `${percentage}%`;
        document.getElementById('progressPercentage').textContent = `${percentage}%`;
        
        // Update steps
        for (let i = 1; i <= 7; i++) {
            const stepElement = document.getElementById(`step${i}`);
            stepElement.classList.remove('active', 'completed');
            
            if (i < step) {
                stepElement.classList.add('completed');
            } else if (i === step) {
                stepElement.classList.add('active');
            }
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    createStatsBar(examData, questionCount) {
        const statsBar = document.createElement('div');
        statsBar.className = 'stats-bar';
        
        // Handle both string and object inputs for backward compatibility
        const title = examData.exam_name || examData.metadata?.title || examData || 'Unknown Paper';
        const paperTitle = examData.metadata?.title || title;
        const citation = examData.metadata?.citation || '';
        const examName = examData.exam_name || title;
        
        const truncatedTitle = paperTitle.length > 60 ? 
            paperTitle.substring(0, 60) + '...' : paperTitle;
        const truncatedExamName = examName.length > 50 ? 
            examName.substring(0, 50) + '...' : examName;
        
        statsBar.innerHTML = `
            <div class="exam-header">
                <div class="stat exam-name">
                    <span class="stat-label">Exam:</span> 
                    <span class="stat-value exam-title">${truncatedExamName}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Questions:</span> 
                    <span class="stat-value">${questionCount}</span>
                </div>
            </div>
            <div class="paper-info">
                <div class="stat">
                    <span class="stat-label">Paper:</span> 
                    <span class="stat-value">${truncatedTitle}</span>
                </div>
                ${citation ? `<div class="citation">
                    <span class="stat-label">Citation:</span> 
                    <span class="citation-text">${citation}</span>
                </div>` : ''}
            </div>
        `;
        return statsBar;
    }

    renderQuestion(question, index) {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question';
        questionDiv.dataset.questionIndex = index;
        questionDiv.dataset.questionType = question.type;
        
        if (question.type === 'multiple_choice') {
            return this.renderMultipleChoiceQuestion(question, index, questionDiv);
        } else if (question.type === 'open_ended') {
            return this.renderOpenEndedQuestion(question, index, questionDiv);
        }
        
        return questionDiv;
    }

    renderMultipleChoiceQuestion(question, index, questionDiv) {
        const choicesHtml = question.choices.map(choice => `
            <label class="choice-label" data-choice="${choice.label}">
                <input type="radio" name="question-${index}" value="${choice.label}">
                <span class="choice-letter">${choice.label}.</span>
                <span class="choice-text">${choice.text}</span>
            </label>
        `).join('');
        
        questionDiv.innerHTML = `
            <div class="question-number">Question ${String(index + 1).padStart(2, '0')}</div>
            <div class="question-type-badge">Multiple Choice</div>
            <div class="question-prompt">${question.prompt}</div>
            <div class="choices">${choicesHtml}</div>
        `;
        
        return questionDiv;
    }

    renderOpenEndedQuestion(question, index, questionDiv) {
        const textareaId = `question-${index}-text`;
        
        questionDiv.innerHTML = `
            <div class="question-number">Question ${String(index + 1).padStart(2, '0')}</div>
            <div class="question-type-badge open-ended">Open Ended</div>
            <div class="question-prompt">${question.prompt}</div>
            <div class="open-ended-input">
                <textarea 
                    id="${textareaId}" 
                    name="question-${index}" 
                    class="open-ended-answer" 
                    placeholder="Type your answer here... Aim for a comprehensive response that addresses the key concepts."
                    maxlength="2000"></textarea>
                <div class="char-count" id="char-count-${index}">0 / 2000 characters</div>
            </div>
        `;
        
        // Add character counter
        const textarea = questionDiv.querySelector(`#${textareaId}`);
        const charCount = questionDiv.querySelector(`#char-count-${index}`);
        
        textarea.addEventListener('input', (e) => {
            const length = e.target.value.length;
            charCount.textContent = `${length} / 2000 characters`;
            charCount.className = length > 1800 ? 'char-count warning' : 'char-count';
        });
        
        return questionDiv;
    }

    createSubmitSection() {
        const submitSection = document.createElement('div');
        submitSection.className = 'submit-section';
        submitSection.innerHTML = `
            <button class="submit-btn" id="submitExamBtn" type="button">
                Submit Exam
            </button>
        `;
        return submitSection;
    }

    collectAnswers() {
        const answers = {};
        this.state.currentQuestions.forEach((question, index) => {
            if (question.type === 'multiple_choice') {
                const selectedInput = document.querySelector(`input[name="question-${index}"]:checked`);
                if (selectedInput) {
                    answers[index] = selectedInput.value;
                }
            } else if (question.type === 'open_ended') {
                const textarea = document.querySelector(`textarea[name="question-${index}"]`);
                if (textarea && textarea.value.trim()) {
                    answers[index] = textarea.value.trim();
                }
            }
        });
        return answers;
    }

    calculateScore(answers) {
        let correct = 0, incorrect = 0, unanswered = 0;
        let mcCorrect = 0, mcTotal = 0, oeAnswered = 0, oeTotal = 0;

        this.state.currentQuestions.forEach((question, index) => {
            const userAnswer = answers[index];
            
            if (question.type === 'multiple_choice') {
                mcTotal++;
                if (!userAnswer) {
                    unanswered++;
                } else if (userAnswer === question.correct) {
                    correct++;
                    mcCorrect++;
                } else {
                    incorrect++;
                }
            } else if (question.type === 'open_ended') {
                oeTotal++;
                if (!userAnswer) {
                    unanswered++;
                } else {
                    // For open-ended questions, just count as "answered"
                    // Actual grading would require more sophisticated analysis
                    oeAnswered++;
                    correct++; // For now, assume answered = correct for scoring
                }
            }
        });

        const total = this.state.currentQuestions.length;
        const percentage = total > 0 ? Math.round((correct / total) * 100) : 0;

        return { 
            correct, incorrect, unanswered, total, percentage,
            mcCorrect, mcTotal, oeAnswered, oeTotal
        };
    }

    showResults(score) {
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'results-container';

        const scoreSummary = document.createElement('div');
        scoreSummary.className = 'score-summary';
        
        // Create detailed breakdown for mixed question types
        let breakdownHtml = `
            <div class="score-item">
                <span class="score-value correct">${score.correct}</span>
                <span>Answered</span>
            </div>
            <div class="score-item">
                <span class="score-value unanswered">${score.unanswered}</span>
                <span>Unanswered</span>
            </div>
        `;
        
        if (score.mcTotal > 0) {
            breakdownHtml += `
                <div class="score-item">
                    <span class="score-value correct">${score.mcCorrect}/${score.mcTotal}</span>
                    <span>MC Correct</span>
                </div>
            `;
        }
        
        if (score.oeTotal > 0) {
            breakdownHtml += `
                <div class="score-item">
                    <span class="score-value correct">${score.oeAnswered}/${score.oeTotal}</span>
                    <span>Open-Ended</span>
                </div>
            `;
        }
        
        scoreSummary.innerHTML = `
            <div class="score-display">${score.percentage}%</div>
            <div class="score-label">Completion Score</div>
            <div class="score-breakdown">${breakdownHtml}</div>
            <button class="restart-btn" onclick="location.reload()">Take Another Exam</button>
        `;
        resultsContainer.appendChild(scoreSummary);

        // Insert at top of exam container
        const firstChild = this.elements.examContainer.firstChild;
        this.elements.examContainer.insertBefore(resultsContainer, firstChild);

        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    showAnswerFeedback(answers) {
        this.state.currentQuestions.forEach((question, index) => {
            const questionDiv = document.querySelector(`[data-question-index="${index}"]`);
            const userAnswer = answers[index];

            if (question.type === 'multiple_choice') {
                this.showMultipleChoiceFeedback(questionDiv, question, userAnswer, index);
            } else if (question.type === 'open_ended') {
                this.showOpenEndedFeedback(questionDiv, question, userAnswer, index);
            }
        });
    }

    showMultipleChoiceFeedback(questionDiv, question, userAnswer, index) {
        const correctAnswer = question.correct;

        // Add question status class
        if (!userAnswer) {
            questionDiv.classList.add('unanswered');
        } else if (userAnswer === correctAnswer) {
            questionDiv.classList.add('answered-correct');
        } else {
            questionDiv.classList.add('answered-incorrect');
        }

        // Mark choices and disable inputs
        const choiceLabels = questionDiv.querySelectorAll('.choice-label');
        choiceLabels.forEach(label => {
            const choiceValue = label.dataset.choice;
            const input = label.querySelector('input');
            input.disabled = true;

            if (choiceValue === correctAnswer) {
                label.classList.add('correct-answer');
            }

            if (userAnswer && choiceValue === userAnswer && userAnswer !== correctAnswer) {
                label.classList.add('user-answer', 'incorrect');
            }
        });

        // Add explanation
        const explanation = document.createElement('div');
        explanation.className = 'feedback-explanation';
        
        let feedbackText = `<strong>Correct Answer:</strong> ${correctAnswer}`;
        if (userAnswer) {
            feedbackText += userAnswer === correctAnswer ? 
                '<br><strong>Your answer was correct!</strong>' : 
                `<br><strong>Your answer (${userAnswer}) was incorrect.</strong>`;
        } else {
            feedbackText += '<br><strong>You did not answer this question.</strong>';
        }
        
        explanation.innerHTML = feedbackText;
        questionDiv.appendChild(explanation);
    }

    showOpenEndedFeedback(questionDiv, question, userAnswer, index) {
        // Add question status class
        if (!userAnswer) {
            questionDiv.classList.add('unanswered');
        } else {
            questionDiv.classList.add('answered-correct'); // Assume answered = good for now
        }

        // Disable textarea
        const textarea = questionDiv.querySelector('textarea');
        if (textarea) {
            textarea.disabled = true;
        }

        // Show sample answer and key points
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-explanation';
        
        let feedbackHtml = '';
        if (!userAnswer) {
            feedbackHtml += '<strong>You did not answer this question.</strong><br><br>';
        } else {
            feedbackHtml += `<strong>Your Response:</strong> Submitted (${userAnswer.length} characters)<br><br>`;
        }
        
        // Add sample answer
        feedbackHtml += `
            <div class="sample-answer">
                <div class="sample-answer-label">Model Answer:</div>
                <div>${question.sample_answer}</div>
            </div>
        `;
        
        // Add key points if available
        if (question.key_points && question.key_points.length > 0) {
            feedbackHtml += `
                <div class="key-points">
                    <div class="key-points-label">Key Points to Address:</div>
                    <ul class="key-points-list">
                        ${question.key_points.map(point => `<li>${point}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        feedbackDiv.innerHTML = feedbackHtml;
        questionDiv.appendChild(feedbackDiv);
    }

    submitExam() {
        if (this.state.examSubmitted) return;

        const answers = this.collectAnswers();
        const score = this.calculateScore(answers);

        this.state.examSubmitted = true;

        // Hide submit button
        const submitBtn = document.getElementById('submitExamBtn');
        if (submitBtn) submitBtn.style.display = 'none';

        // Show feedback and results
        this.showAnswerFeedback(answers);
        this.showResults(score);
    }

    async generateExam() {
        const arxivId = this.elements.arxivInput.value.trim();
        
        // Reset state
        this.state.currentQuestions = [];
        this.state.examSubmitted = false;
        this.elements.examContainer.innerHTML = '';
        this.hideError();
        this.elements.downloadExamBtn.style.display = 'none'; // Hide download button on new generation

        if (!arxivId) {
            this.showError('Please enter an arXiv paper ID.');
            return;
        }

        if (!this.validateArxivId(arxivId)) {
            this.showError('Invalid arXiv ID format. Use: 2406.01234, hep-th/0401001, or math.AG/0601001');
            return;
        }

        this.showLoading();
        this.updateProgressSteps(false, 7, 3); // Single paper mode

        try {
            // Step 1: Fetching paper content
            this.updateProgress(1, 10);
            await this.delay(400);
            
            // Step 2: Analyzing document structure
            this.updateProgress(2, 20);
            await this.delay(400);
            
            // Step 3: Processing with AI 
            this.updateProgress(3, 30);
            await this.delay(400);
            
            // Step 4: Generating multiple-choice questions
            this.updateProgress(4, 50);
            
            // Validate API key
            if (!this.state.llmConfig.apiKey) {
                throw new Error('Please configure your API key in the LLM Configuration section above.');
            }

            // Start the request
            const responsePromise = fetch(`/api/exam/${arxivId}?mc_questions=7&oe_questions=3&teacher_mode=${this.state.teacherMode}`, {
                method: 'GET',
                headers: {
                    'X-LLM-Provider': this.state.llmConfig.provider,
                    'X-LLM-Model': this.state.llmConfig.model,
                    'X-LLM-API-Key': this.state.llmConfig.apiKey
                }
            });
            
            // Simulate MC question generation progress
            await this.delay(800);
            this.updateProgress(4, 65);
            await this.delay(400);
            
            // Step 5: Generating open-ended questions
            this.updateProgress(5, 80);
            await this.delay(600);
            
            // Wait for the actual response
            const response = await responsePromise;
            const data = await response.json();
            
            // Step 6: Finalizing exam
            this.updateProgress(6, 90);
            await this.delay(400);

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate exam');
            }

            if (!data || !data.questions || data.questions.length === 0) {
                throw new Error('No questions generated. Paper might lack sufficient content.');
            }
            
            // Step 7: Naming exam
            this.updateProgress(7, 100);
            await this.delay(500);

            // Store questions and full exam data 
            this.state.currentQuestions = data.questions; // Access questions from data.questions
            this.state.currentExamData = data; // Store full exam data for download
            console.log(`Received ${data.questions.length} questions from API:`, data.questions);
            
            // Clear exam container and add stats bar with full exam data
            this.elements.examContainer.innerHTML = '';
            this.elements.examContainer.appendChild(this.createStatsBar(data, data.questions.length));
            
            data.questions.forEach((question, index) => { // Iterate over data.questions
                this.elements.examContainer.appendChild(this.renderQuestion(question, index));
            });

            this.elements.examContainer.appendChild(this.createSubmitSection());

            // Bind submit handler (only if not in teacher mode)
            if (!this.state.teacherMode) {
                const submitBtn = document.getElementById('submitExamBtn');
                if (submitBtn) {
                    submitBtn.addEventListener('click', () => this.submitExam());
                }
            }

            // Complete progress bar (already at 100% from step 6)
            await this.delay(500);

            // Show teacher answers if in teacher mode
            if (this.state.teacherMode) {
                this.showAllAnswers();
            }

            this.elements.examContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            this.elements.downloadExamBtn.style.display = 'block'; // Show download button after generation

        } catch (error) {
            console.error('Exam generation error:', error);
            this.showError(error.message || 'Failed to generate exam. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    createTeacherAnswers(question) {
        const teacherDiv = document.createElement('div');
        teacherDiv.className = 'teacher-answers';
        
        let content = '<div class="teacher-answers-header">Teacher Mode - All Answers</div>';
        
        if (question.type === 'multiple_choice') {
            content += this.createMCTeacherAnswer(question);
        } else if (question.type === 'open_ended') {
            content += this.createOETeacherAnswer(question);
        }
        
        teacherDiv.innerHTML = content;
        return teacherDiv;
    }

    createMCTeacherAnswer(question) {
        const correctChoice = question.choices.find(c => c.label === question.correct);
        return `
            <div class="mc-correct-answer">
                <div class="mc-correct-label">Correct Answer:</div>
                <strong>${question.correct}. ${correctChoice ? correctChoice.text : 'Unknown'}</strong>
            </div>
        `;
    }

    createOETeacherAnswer(question) {
        let content = `
            <div class="sample-answer" style="margin-bottom: 20px;">
                <div class="sample-answer-label">Model Answer:</div>
                <div>${question.sample_answer}</div>
            </div>
        `;
        
        if (question.graded_examples && question.graded_examples.length > 0) {
            content += '<div class="graded-examples">';
            
            // Sort examples by score (5 to 1)
            const sortedExamples = [...question.graded_examples].sort((a, b) => b.score - a.score);
            
            sortedExamples.forEach(example => {
                content += `
                    <div class="graded-example">
                        <div class="example-header">
                            <span class="example-score score-${example.score}">${example.score}/5 Points</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--graphite);">
                                ${this.getScoreLabel(example.score)}
                            </span>
                        </div>
                        <div class="example-content">
                            <div class="example-answer">${example.answer}</div>
                            <div class="example-feedback">
                                <strong>Feedback:</strong> ${example.feedback}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            content += '</div>';
        }
        
        if (question.key_points && question.key_points.length > 0) {
            content += `
                <div class="key-points" style="margin-top: 16px;">
                    <div class="key-points-label">Key Points to Address:</div>
                    <ul class="key-points-list">
                        ${question.key_points.map(point => `<li>${point}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        return content;
    }

    getScoreLabel(score) {
        const labels = {
            5: 'Excellent',
            4: 'Good', 
            3: 'Satisfactory'
        };
        return labels[score] || 'Unknown';
    }

    // Search functionality
    initializeSearch() {
        // Mode toggle event listeners
        this.elements.modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });

        // Search tab event listeners
        this.elements.searchTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchSearchType(e.target.dataset.search);
            });
        });

        // Search button listeners
        this.elements.performSearchBtn.addEventListener('click', () => {
            this.performSearch();
        });

        this.elements.generateFromSearchBtn.addEventListener('click', () => {
            this.generateExamFromSearch();
        });

        // Enter key support for search inputs
        [this.elements.generalQuery, this.elements.authorQuery].forEach(input => {
            if (input) {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.performSearch();
                    }
                });
            }
        });
    }

    switchMode(mode) {
        this.state.currentMode = mode;
        
        // Update button states
        this.elements.modeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Show/hide appropriate content
        this.elements.singlePaperMode.style.display = mode === 'single' ? 'block' : 'none';
        this.elements.searchMode.style.display = mode === 'search' ? 'block' : 'none';
        this.elements.examGenerationControls.style.display = mode === 'single' ? 'block' : 'none';
        this.elements.searchInput.style.display = mode === 'search' ? 'block' : 'none';

        // Clear any existing content
        this.clearResults();
    }

    switchSearchType(searchType) {
        this.state.currentSearchType = searchType;
        
        // Update tab states
        this.elements.searchTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.search === searchType);
        });

        // Show/hide appropriate panels
        this.elements.searchPanels.forEach(panel => {
            panel.classList.toggle('active', panel.id === searchType + 'Search');
        });

        // Clear results when switching search types
        this.clearResults();
    }

    async performSearch() {
        const searchType = this.state.currentSearchType;
        let query, sortBy, endpoint;

        try {
            // Build search parameters based on type
            switch (searchType) {
                case 'general':
                    query = this.elements.generalQuery.value.trim();
                    sortBy = this.elements.generalSort.value;
                    endpoint = `/api/search?q=${encodeURIComponent(query)}&sort_by=${sortBy}&max_results=20`;
                    break;
                
                case 'author':
                    query = this.elements.authorQuery.value.trim();
                    sortBy = this.elements.authorSort.value;
                    endpoint = `/api/search/author/${encodeURIComponent(query)}?sort_by=${sortBy}&max_results=20`;
                    break;
                
                case 'category':
                    const category = this.elements.categorySelect.value;
                    if (!category) {
                        this.showError('Please select a category');
                        return;
                    }
                    sortBy = this.elements.categorySort.value;
                    endpoint = `/api/search/category/${encodeURIComponent(category)}?sort_by=${sortBy}&max_results=20`;
                    
                    // Add date filters if provided
                    const dateFrom = this.elements.dateFrom.value;
                    const dateTo = this.elements.dateTo.value;
                    if (dateFrom) endpoint += `&date_from=${dateFrom}`;
                    if (dateTo) endpoint += `&date_to=${dateTo}`;
                    break;
                
                default:
                    this.showError('Invalid search type');
                    return;
            }

            if (!query && searchType !== 'category') {
                this.showError('Please enter a search query');
                return;
            }

            // Show loading state
            this.elements.performSearchBtn.disabled = true;
            this.elements.performSearchBtn.textContent = 'Searching...';

            // Perform search
            const response = await fetch(endpoint);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Search failed');
            }

            const data = await response.json();
            this.state.searchResults = data.results;
            this.displaySearchResults(data);

        } catch (error) {
            console.error('Search error:', error);
            this.showError(error.message || 'Search failed. Please try again.');
        } finally {
            // Reset button state
            this.elements.performSearchBtn.disabled = false;
            this.elements.performSearchBtn.textContent = 'Search Papers';
        }
    }

    displaySearchResults(data) {
        if (!data.results || data.results.length === 0) {
            this.elements.resultsList.innerHTML = '<div class="no-results">No papers found. Try a different search query.</div>';
            this.elements.searchResults.style.display = 'block';
            return;
        }

        let html = '';
        data.results.forEach((result, index) => {
            html += `
                <div class="result-item" data-index="${index}" data-arxiv-id="${result.arxiv_id}">
                    <div class="result-title">${result.title}</div>
                    <div class="result-authors">${result.authors.join(', ')}</div>
                    <div class="result-abstract">${result.abstract.substring(0, 300)}${result.abstract.length > 300 ? '...' : ''}</div>
                    <div class="result-meta">
                        <div class="result-categories">
                            ${result.categories.map(cat => `<span class="category-tag">${cat}</span>`).join('')}
                        </div>
                        <div class="result-date">${result.published ? new Date(result.published).toLocaleDateString() : ''}</div>
                    </div>
                    <div class="result-actions">
                        <button class="download-pdf-btn" data-arxiv-id="${result.arxiv_id}">
                            📄 Download PDF
                        </button>
                    </div>
                </div>
            `;
        });

        this.elements.resultsList.innerHTML = html;
        this.elements.searchResults.style.display = 'block';

        // Add click listeners to result items
        this.elements.resultsList.querySelectorAll('.result-item').forEach(item => {
            item.addEventListener('click', () => {
                this.togglePaperSelection(item);
            });
        });

        // Add click listeners to download buttons
        this.elements.resultsList.querySelectorAll('.download-pdf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering result item click
                const arxivId = btn.getAttribute('data-arxiv-id');
                if (arxivId) {
                    this.downloadPDF(arxivId);
                }
            });
        });

        // Papers are not auto-selected - users must manually select them
    }

    togglePaperSelection(item, forceSelect = false) {
        const arxivId = item.dataset.arxivId;
        const maxPapers = parseInt(this.elements.maxPapers.value);
        
        if (item.classList.contains('selected') && !forceSelect) {
            // Deselect
            item.classList.remove('selected');
            this.state.selectedPapers = this.state.selectedPapers.filter(p => p.arxiv_id !== arxivId);
        } else if (this.state.selectedPapers.length < maxPapers || forceSelect) {
            // Select
            item.classList.add('selected');
            const result = this.state.searchResults.find(r => r.arxiv_id === arxivId);
            if (result && !this.state.selectedPapers.find(p => p.arxiv_id === arxivId)) {
                this.state.selectedPapers.push(result);
            }
        } else {
            this.showError(`Maximum ${maxPapers} papers can be selected for exam generation.`);
        }

        // Update generate button state and text with count
        const selectedCount = this.state.selectedPapers.length;
        this.elements.generateFromSearchBtn.disabled = selectedCount === 0;
        this.elements.generateFromSearchBtn.textContent = `Generate Exam from Selected (${selectedCount})`;
    }

    async generateExamFromSearch() {
        if (this.state.selectedPapers.length === 0) {
            this.showError('Please select at least one paper');
            return;
        }

        if (!this.state.llmConfig.apiKey) {
            this.showError('Please configure your LLM API key first');
            return;
        }

        // Extract arXiv IDs from selected papers
        const arxivIds = this.state.selectedPapers.map(paper => paper.arxiv_id);
        
        // Validate arXiv IDs
        const invalidIds = arxivIds.filter(id => !id || !id.match(/^\d{4}\.\d{4,5}$/));
        if (invalidIds.length > 0) {
            this.showError(`Invalid arXiv IDs detected: ${invalidIds.join(', ')}. Please try searching again.`);
            return;
        }
        
        const payload = {
            arxiv_ids: arxivIds,
            mc_questions: parseInt(this.elements.mcQuestions.value),
            oe_questions: parseInt(this.elements.oeQuestions.value),
            exam_title: `Multi-Paper Exam: ${this.getCurrentSearchQuery()}`
        };

        const headers = {
            'Content-Type': 'application/json',
            'X-LLM-Provider': this.state.llmConfig.provider,
            'X-LLM-Model': this.state.llmConfig.model,
            'X-LLM-API-Key': this.state.llmConfig.apiKey
        };

        try {
            this.showLoading();
            this.updateProgressSteps(true, parseInt(this.elements.mcQuestions.value), parseInt(this.elements.oeQuestions.value)); // Multi-paper mode
            this.elements.generateFromSearchBtn.disabled = true;
            this.elements.generateFromSearchBtn.textContent = 'Generating Exam...';

            // Multi-paper exam progress steps
            // Step 1: Processing selected papers
            this.updateProgress(1, 10);
            await this.delay(300);
            
            // Step 2: Downloading and analyzing papers
            this.updateProgress(2, 25);
            await this.delay(400);
            
            // Step 3: Combining content from multiple papers
            this.updateProgress(3, 40);
            await this.delay(300);

            // Start the request
            const responsePromise = fetch('/api/exam/selected', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload)
            });
            
            // Step 4: Generating multiple-choice questions
            this.updateProgress(4, 55);
            await this.delay(800);
            this.updateProgress(4, 70);
            await this.delay(400);
            
            // Step 5: Generating open-ended questions  
            this.updateProgress(5, 85);
            await this.delay(600);

            // Wait for the actual response
            const response = await responsePromise;
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to generate exam');
            }

            const examData = await response.json();
            
            // Step 6: Finalizing multi-paper exam
            this.updateProgress(6, 90);
            await this.delay(400);
            
            // Display the exam (same logic as generateExam)
            if (!examData || !examData.questions || examData.questions.length === 0) {
                throw new Error('No questions generated. Papers might lack sufficient content.');
            }
            
            // Step 7: Naming exam
            this.updateProgress(7, 100);
            await this.delay(500);

            // Store questions and full exam data
            this.state.currentQuestions = examData.questions;
            this.state.currentExamData = examData; // Store full exam data for download
            console.log(`Received ${examData.questions.length} questions from multi-paper exam:`, examData.questions);
            
            const examTitle = examData.metadata.title || 'Multi-Paper Exam';
            
            // Clear and populate exam container
            this.elements.examContainer.innerHTML = '';
            this.elements.examContainer.appendChild(this.createStatsBar(examData, examData.questions.length));
            
            examData.questions.forEach((question, index) => {
                this.elements.examContainer.appendChild(this.renderQuestion(question, index));
            });

            this.elements.examContainer.appendChild(this.createSubmitSection());

            // Bind submit handler (only if not in teacher mode)
            if (!this.state.teacherMode) {
                const submitBtn = document.getElementById('submitExamBtn');
                if (submitBtn) {
                    submitBtn.addEventListener('click', () => this.submitExam());
                }
            }

            // Show teacher answers if in teacher mode
            if (this.state.teacherMode) {
                this.showAllAnswers();
            }
            
            // Show download button
            this.elements.downloadExamBtn.style.display = 'block';

        } catch (error) {
            console.error('Exam generation error:', error);
            
            let errorMessage = error.message || 'Failed to generate exam. Please try again.';
            
            // Provide more helpful error messages based on common issues
            if (errorMessage.includes('Failed to process any papers')) {
                errorMessage += ' This might be due to invalid paper IDs or network issues. Try selecting different papers.';
            } else if (errorMessage.includes('404')) {
                errorMessage = 'Selected papers could not be found. Please try searching again and selecting different papers.';
            } else if (errorMessage.includes('API key')) {
                errorMessage = 'Please configure your LLM API key in the settings above.';
            }
            
            this.showError(errorMessage);
        } finally {
            this.hideLoading();
            this.elements.generateFromSearchBtn.disabled = false;
            this.elements.generateFromSearchBtn.textContent = 'Generate Exam from Selected';
        }
    }

    getCurrentSearchQuery() {
        switch (this.state.currentSearchType) {
            case 'general':
                return this.elements.generalQuery.value.trim();
            case 'author':
                return `au:${this.elements.authorQuery.value.trim()}`;
            case 'category':
                return `cat:${this.elements.categorySelect.value}`;
            default:
                return '';
        }
    }

    getCurrentSortBy() {
        switch (this.state.currentSearchType) {
            case 'general':
                return this.elements.generalSort.value;
            case 'author':
                return this.elements.authorSort.value;
            case 'category':
                return this.elements.categorySort.value;
            default:
                return 'relevance';
        }
    }

    clearResults() {
        this.elements.searchResults.style.display = 'none';
        this.elements.resultsList.innerHTML = '';
        this.state.searchResults = [];
        this.state.selectedPapers = [];
        this.clearExam();
    }

    clearExam() {
        this.elements.examContainer.innerHTML = '';
        this.elements.errorContainer.innerHTML = '';
        this.elements.downloadExamBtn.style.display = 'none';
        this.state.currentQuestions = [];
        this.state.currentExamData = null;
        this.state.examSubmitted = false;
    }

    downloadEnhancedExam() {
        try {
            const examData = this.state.currentExamData;
            const questions = this.state.currentQuestions;
            
            if (!examData || !questions) {
                console.error('No exam data available for download');
                return;
            }

            // Create comprehensive exam export with metadata
            const exportData = {
                // Exam metadata
                exam_info: {
                    exam_name: examData.exam_name || 'arXiv Exam',
                    generated_date: new Date().toISOString(),
                    generated_by: 'arXiv IQ - Research Paper Exam Generator',
                    website: 'https://github.com/jcdavis131/arxiv-exam-app',
                    llm_provider: this.state.llmConfig.provider || 'openai',
                    llm_model: this.state.llmConfig.model || 'gpt-4o-mini',
                    total_questions: questions.length,
                    question_breakdown: {
                        multiple_choice: questions.filter(q => q.type === 'multiple_choice').length,
                        open_ended: questions.filter(q => q.type === 'open_ended').length
                    }
                },
                
                // Paper metadata and citation
                source_paper: {
                    title: examData.metadata.title,
                    authors: examData.metadata.authors,
                    arxiv_id: examData.metadata.arxiv_id,
                    categories: examData.metadata.categories,
                    published: examData.metadata.published,
                    abstract: examData.metadata.abstract,
                    citation: examData.metadata.citation
                },
                
                // Full questions array
                questions: questions,
                
                // Export metadata
                export_info: {
                    format_version: '1.0',
                    exported_at: new Date().toISOString(),
                    export_type: 'comprehensive_exam_data'
                }
            };

            // Create filename based on exam name and paper ID
            const paperIdentifier = examData.metadata.arxiv_id || 
                                   this.elements.arxivInput.value.trim() || 
                                   'exam';
            const examName = examData.exam_name ? 
                           examData.exam_name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase() : 
                           'arxiv_exam';
            const timestamp = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
            const filename = `${examName}_${paperIdentifier}_${timestamp}.json`;

            // Download the enhanced JSON
            const dataStr = JSON.stringify(exportData, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            console.log(`Downloaded enhanced exam data as ${filename}`);

        } catch (error) {
            console.error('Error downloading exam data:', error);
            alert('Failed to download exam data. Please try again.');
        }
    }

    async downloadPDF(arxivId) {
        try {
            console.log(`Downloading PDF for ${arxivId}`);
            
            // Show loading state on the button
            const downloadBtn = document.querySelector(`[data-arxiv-id="${arxivId}"].download-pdf-btn`);
            if (downloadBtn) {
                downloadBtn.disabled = true;
                downloadBtn.textContent = '⏳ Downloading...';
            }
            
            // Make request to download endpoint
            const response = await fetch(`/api/download/${arxivId}`);
            
            if (!response.ok) {
                throw new Error(`Download failed: ${response.status} ${response.statusText}`);
            }
            
            // Get the filename from response headers
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `${arxivId}.pdf`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            // Create blob and download
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            
            // Create temporary download link
            const downloadLink = document.createElement('a');
            downloadLink.href = downloadUrl;
            downloadLink.download = filename;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // Clean up object URL
            window.URL.revokeObjectURL(downloadUrl);
            
            console.log(`Successfully downloaded ${filename}`);
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError(`Failed to download PDF: ${error.message}`);
        } finally {
            // Reset button state
            const downloadBtn = document.querySelector(`[data-arxiv-id="${arxivId}"].download-pdf-btn`);
            if (downloadBtn) {
                downloadBtn.disabled = false;
                downloadBtn.textContent = '📄 Download PDF';
            }
        }
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ExamApp();
});