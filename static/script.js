// arXiv Exam Generator - Client-side functionality
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
            downloadExamBtn: document.getElementById('downloadExamBtn')
        };
        
        this.state = {
            currentQuestions: [],
            examSubmitted: false,
            teacherMode: false,
            llmConfig: this.loadConfig()
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
        
        // LLM Configuration events
        this.elements.llmProvider.addEventListener('change', () => this.updateModelOptions());
        this.elements.saveConfig.addEventListener('click', () => this.saveConfig());
        this.elements.apiKey.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.saveConfig();
        });

        // Download Exam events
        this.elements.downloadExamBtn.addEventListener('click', () => {
            if (this.state.currentQuestions.length > 0) {
                const dataStr = JSON.stringify(this.state.currentQuestions, null, 2);
                const blob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `exam_${this.elements.arxivInput.value.trim()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
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

    resetProgressBar() {
        // Reset all progress elements
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressPercentage').textContent = '0%';
        
        // Reset all steps
        for (let i = 1; i <= 6; i++) {
            const step = document.getElementById(`step${i}`);
            step.classList.remove('active', 'completed');
        }
    }

    updateProgress(step, percentage) {
        // Update percentage and fill
        document.getElementById('progressFill').style.width = `${percentage}%`;
        document.getElementById('progressPercentage').textContent = `${percentage}%`;
        
        // Update steps
        for (let i = 1; i <= 6; i++) {
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

    createStatsBar(paperTitle, questionCount) {
        const statsBar = document.createElement('div');
        statsBar.className = 'stats-bar';
        const truncatedTitle = paperTitle.length > 60 ? 
            paperTitle.substring(0, 60) + '...' : paperTitle;
        
        statsBar.innerHTML = `
            <div class="stat">
                <span class="stat-label">Paper:</span> 
                <span class="stat-value">${truncatedTitle}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Questions:</span> 
                <span class="stat-value">${questionCount}</span>
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
                <div class="sample-answer-label">Sample Answer:</div>
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
            
            // Step 4: Generating 10 multiple-choice questions
            this.updateProgress(4, 50);
            
            // Validate API key
            if (!this.state.llmConfig.apiKey) {
                throw new Error('Please configure your API key in the LLM Configuration section above.');
            }

            // Request split generation: 7 MC + 3 OE questions
            const response = await fetch(`/exam/${arxivId}?mc_questions=7&oe_questions=3&teacher_mode=false`, {
                method: 'GET',
                headers: {
                    'X-LLM-Provider': this.state.llmConfig.provider,
                    'X-LLM-Model': this.state.llmConfig.model,
                    'X-LLM-API-Key': this.state.llmConfig.apiKey
                }
            });
            
            // Step 5: Generating 5 open-ended questions (this happens during the fetch)
            this.updateProgress(5, 80);
            
            const data = await response.json();
            
            // Step 6: Finalizing exam
            this.updateProgress(6, 100);

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate exam');
            }

            if (!data || !data.questions || data.questions.length === 0) {
                throw new Error('No questions generated. Paper might lack sufficient content.');
            }

            // Store questions and render exam
            this.state.currentQuestions = data.questions; // Access questions from data.questions
            console.log(`Received ${data.questions.length} questions from API:`, data.questions);
            
            const paperTitle = data.metadata.title || 'Unknown Paper'; // Access title from data.metadata.title
            
            this.elements.examContainer.appendChild(this.createStatsBar(paperTitle, data.questions.length));
            
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
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ExamApp();
});