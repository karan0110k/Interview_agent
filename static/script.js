document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const setupView = document.getElementById('setup-view');
    const interviewView = document.getElementById('interview-view');
    const interviewForm = document.getElementById('interview-form');
    const startBtn = document.getElementById('start-btn');
    const endBtn = document.getElementById('end-btn');
    const retryBtn = document.getElementById('retry-btn');
    const summaryBtn = document.getElementById('summary-btn');
    const statusText = document.getElementById('status-text');
    const transcriptDiv = document.getElementById('transcript');
    const networkStatus = document.getElementById('network-status');
    const networkStatusText = document.getElementById('network-status-text');
    const statusDot = document.querySelector('.status-dot');
    const fileUpload = document.getElementById('resume-upload');
    const summaryModal = document.getElementById('summary-modal');
    const closeModalBtn = document.querySelector('.close-modal');
    
    // Check if we're on the interview page
    const isInterviewPage = setupView !== null && interviewView !== null;

    // --- State & Speech API ---
    let sessionId = null;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;
    let isListening = false;
    let retryCount = 0;
    const MAX_RETRIES = 5; // Maximum number of consecutive retries

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        console.log("Speech Recognition API initialized.");
    } else {
        console.error("Web Speech API not supported in this browser.");
        alert("Sorry, your browser does not support the Web Speech API. Please try Chrome or Firefox.");
        startBtn.disabled = true;
    }

    // Check if network is online
    function isNetworkOnline() {
        return navigator.onLine;
    }
    
    // Update network status indicator
    function updateNetworkStatus() {
        const online = isNetworkOnline();
        networkStatus.className = online ? 'network-status online' : 'network-status offline';
        networkStatusText.textContent = online ? 'Online' : 'Offline';
        
        // Show/hide retry button based on network status and session state
        if (!online && sessionId) {
            retryBtn.classList.remove('hidden');
        } else {
            retryBtn.classList.add('hidden');
        }
    }
    
    function startSpeechRecognition() {
        if (!isNetworkOnline()) {
            console.error("Network is offline. Cannot start speech recognition.");
            updateStatus("Network is offline. Please check your internet connection and try again.");
            // Try again after a delay if network comes back online
            setTimeout(() => {
                if (isNetworkOnline()) {
                    console.log("Network is back online. Attempting to restart speech recognition...");
                    startSpeechRecognition();
                }
            }, 3000);
            return;
        }
        
        if (recognition && !isListening) {
            console.log("Attempting to start speech recognition...");
            try {
                recognition.start();
            } catch (e) {
                console.error("Error starting speech recognition:", e);
                updateStatus("Error: Could not start microphone. Please check permissions.");
                retryCount++;
            }
        } else if (isListening) {
            console.log("Speech recognition is already active.");
        }
    }

    // --- Event Listeners ---
    if (isInterviewPage) {
        interviewForm.addEventListener('submit', startInterview);
        endBtn.addEventListener('click', endInterview);
        retryBtn.addEventListener('click', handleReconnection);
        summaryBtn.addEventListener('click', showInterviewSummary);
        closeModalBtn.addEventListener('click', closeModal);
        
        // Initial network status check
        updateNetworkStatus();
        
        // Network status event listeners
        window.addEventListener('online', () => {
            console.log('Network is now online');
            updateStatus('Network connection restored. Continuing interview...');
            updateNetworkStatus();
        });
        
        window.addEventListener('offline', () => {
              console.log('Network is now offline');
              updateStatus('Network connection lost. Please check your internet connection.');
              updateNetworkStatus();
              
              // If we're listening, stop the recognition
              if (recognition && isListening) {
                  try {
                      recognition.stop();
                      isListening = false;
                  } catch (e) {
                      console.error('Error stopping recognition:', e);
                  }
              }
          });
    }
    
    // Network status event listeners for non-interview pages
    if (!isInterviewPage) {
        // Add any global event listeners here if needed
        // Use the reconnection handler for more robust recovery
        if (sessionId) {
            handleReconnection();
        }
    }
    
    // Initialize metric score styling for session details page
    const metricScores = document.querySelectorAll('.metric-score');
    if (metricScores.length > 0) {
        metricScores.forEach(score => {
            const scoreValue = parseFloat(score.getAttribute('data-score'));
            if (scoreValue >= 0.8) {
                score.style.color = '#22c55e'; // Green for high scores
            } else if (scoreValue >= 0.6) {
                score.style.color = '#f59e0b'; // Orange for medium scores
            } else {
                score.style.color = '#ef4444'; // Red for low scores
            }
        });
    }
    
    // End of DOMContentLoaded event listener
    
    // Check network status periodically
    setInterval(updateNetworkStatus, 5000);

    if (recognition) {
        recognition.onstart = () => {
            isListening = true;
            console.log("Event: recognition.onstart - Speech recognition started successfully.");
            updateStatus("Listening for your answer...");
        };

        recognition.onresult = (event) => {
            const userAnswer = event.results[0][0].transcript;
            console.log(`Event: recognition.onresult - Recognized: "${userAnswer}"`);
            updateStatus("Processing your answer...");
            addMessageToTranscript('You', userAnswer);
            // Reset retry counter on successful recognition
            retryCount = 0;
            sendAnswerToServer(userAnswer);
        };

        recognition.onerror = (event) => {
            console.error(`Event: recognition.onerror - Error: ${event.error}`);
            isListening = false;
            let errorMessage = `An error occurred: ${event.error}`;
            let retryDelay = 500; // Default retry delay in ms
            let shouldRetry = false;
            
            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                errorMessage = "Error: Microphone access denied. Please enable it in your browser settings and refresh the page.";
                shouldRetry = false;
            } else if (event.error === 'no-speech') {
                errorMessage = "No speech was detected. I'm listening again.";
                shouldRetry = true;
            } else if (event.error === 'audio-capture') {
                errorMessage = "Error: There was a problem with your microphone. Please check your hardware.";
                shouldRetry = false;
            } else if (event.error === 'network') {
                errorMessage = "Network error occurred. Retrying in a moment...";
                retryDelay = 1500; // Longer delay for network issues
                shouldRetry = true;
            } else if (event.error === 'aborted') {
                errorMessage = "Speech recognition was aborted. Restarting...";
                shouldRetry = true;
            } else {
                // For any other errors, attempt to retry
                errorMessage = `Error: ${event.error}. Attempting to restart...`;
                retryDelay = 1000;
                shouldRetry = true;
            }
            
            updateStatus(errorMessage);
            
            // Implement retry logic with maximum retry limit
            if (shouldRetry) {
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    console.log(`Will retry speech recognition in ${retryDelay}ms (Attempt ${retryCount} of ${MAX_RETRIES})`);
                    setTimeout(startSpeechRecognition, retryDelay);
                } else {
                    console.error(`Maximum retry attempts (${MAX_RETRIES}) reached. Please try again manually.`);
                    updateStatus(`Too many errors occurred. Please click 'End Interview' and try again.`);
                    retryCount = 0; // Reset for next session
                }
            }
        };

        recognition.onend = () => {
            isListening = false;
            console.log("Event: recognition.onend - Speech recognition ended.");
        };
    }

    // --- Core Functions ---
    async function startInterview(e) {
        e.preventDefault();
        
        // Show loading state
        setButtonLoading(startBtn, true);
        updateStatus('Uploading and analyzing your resume...');
        statusDot.style.color = '#fbbf24'; // Amber color during processing

        try {
            const formData = new FormData(interviewForm);
            const response = await fetch('/start-interview', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to start interview.');

            sessionId = data.session_id;
            
            // Smooth transition to interview view
            setupView.style.opacity = '0';
            setTimeout(() => {
                setupView.classList.add('hidden');
                interviewView.classList.remove('hidden');
                // Fade in the interview view
                setTimeout(() => {
                    interviewView.style.opacity = '1';
                }, 50);
            }, 300);

            statusDot.style.color = '#10b981'; // Green for active status
            askQuestion(data.question);

        } catch (error) {
            console.error('Error starting interview:', error);
            // Show error in a more user-friendly way
            updateStatus(`Error: ${error.message}`);
            statusDot.style.color = '#ef4444'; // Red for error
            setTimeout(() => {
                resetUI();
            }, 3000);
        }
    }

    function askQuestion(text) {
        updateStatus("Interviewer is asking a question...");
        addMessageToTranscript('Interviewer', text);

        // Add typing animation to the interviewer message
        const lastMessage = transcriptDiv.lastElementChild;
        if (lastMessage) {
            lastMessage.classList.add('typing');
            setTimeout(() => {
                lastMessage.classList.remove('typing');
            }, 1000);
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => {
            updateStatus("Listening for your answer...");
            statusDot.style.color = '#10b981'; // Green for active listening
            statusDot.style.animation = 'pulse 1.5s infinite'; // Pulsing animation
            startSpeechRecognition();
        };
        window.speechSynthesis.speak(utterance);
    }

    // Track recent questions to detect repetition
    const recentQuestions = [];
    const MAX_RECENT_QUESTIONS = 5;
    
    async function sendAnswerToServer(answer) {
        updateStatus('AI is thinking...');
        try {
            const response = await fetch('/next-question', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, answer: answer }),
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to get next question.');
            
            // Check if this question is a repeat of the most recent question
            if (recentQuestions.length > 0 && data.question === recentQuestions[recentQuestions.length - 1]) {
                console.warn('Detected repeated question. Requesting a new one...');
                updateStatus('Detected a repeated question. Requesting a new one...');
                
                // Request a new question by sending a special flag
                const newResponse = await fetch('/next-question', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        session_id: sessionId, 
                        answer: "I've already answered this question. Please ask a different question.",
                        force_new_question: true
                    }),
                });
                
                const newData = await newResponse.json();
                if (!newResponse.ok) throw new Error(newData.error || 'Failed to get a new question.');
                
                askQuestion(newData.question);
                
                // Add to recent questions list
                recentQuestions.push(newData.question);
                if (recentQuestions.length > MAX_RECENT_QUESTIONS) {
                    recentQuestions.shift(); // Remove oldest question
                }
            } else {
                // Not a repeat, proceed normally
                askQuestion(data.question);
                
                // Add to recent questions list
                recentQuestions.push(data.question);
                if (recentQuestions.length > MAX_RECENT_QUESTIONS) {
                    recentQuestions.shift(); // Remove oldest question
                }
            }

        } catch (error) {
            console.error('Error sending answer:', error);
            alert(`An error occurred: ${error.message}`);
            endInterview();
        }
    }

    function endInterview() {
        if (recognition && isListening) {
            try {
                recognition.stop();
            } catch (e) {
                console.error('Error stopping recognition:', e);
            }
        }
        
        // Show ending animation/message
        updateStatus('Interview completed successfully!');
        statusDot.style.color = '#10b981'; // Green
        statusDot.style.animation = 'none';
        
        // Add a final message to transcript
        addMessageToTranscript('Interviewer', 'Thank you for completing this interview session. You can now view your performance summary or start a new interview.');
        
        // Show the summary button
        document.getElementById('summary-btn').classList.remove('hidden');
        
        // Hide the end button
        endBtn.classList.add('hidden');
        
        // Reset counters and state
        retryCount = 0;
        isListening = false;
    }
    
    // Function to handle reconnection attempts
    function handleReconnection() {
        if (!sessionId) return; // No active interview
        
        if (!isNetworkOnline()) {
            updateStatus('Waiting for network connection...');
            // Check again after a delay
            setTimeout(handleReconnection, 5000);
            return;
        }
        
        updateStatus('Reconnecting to interview...');
        // If we have a session but aren't listening, restart speech recognition
        if (!isListening) {
            setTimeout(startSpeechRecognition, 1000);
        }
    }

    // --- UI & Utility Functions ---
    function addMessageToTranscript(sender, message) {
        const messageEl = document.createElement('div');
        messageEl.classList.add('message', sender.toLowerCase());
        
        // Add avatar/icon for the sender
        const icon = sender === 'Interviewer' ? 
            '<i class="fa-solid fa-robot message-avatar"></i>' : 
            '<i class="fa-solid fa-user message-avatar"></i>';
            
        messageEl.innerHTML = `
            ${icon}
            <div class="message-content">
                <div class="message-sender">${sender}</div>
                <div class="message-text">${message}</div>
            </div>
        `;
        
        transcriptDiv.appendChild(messageEl);
        
        // Smooth scroll to the new message
        messageEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    function resetUI() {
        // Fade out interview view
        interviewView.style.opacity = '0';
        
        setTimeout(() => {
            interviewView.classList.add('hidden');
            setupView.classList.remove('hidden');
            
            // Reset button states
            setButtonLoading(startBtn, false);
            endBtn.classList.remove('hidden');
            summaryBtn.classList.add('hidden');
            
            // Remove any added buttons (like summary button)
            const controlsDiv = document.querySelector('.interview-controls');
            while (controlsDiv.firstChild !== endBtn && controlsDiv.firstChild !== retryBtn) {
                controlsDiv.removeChild(controlsDiv.firstChild);
            }
            
            // Clear transcript
            transcriptDiv.innerHTML = '';
            
            // Reset state
            sessionId = null;
            retryBtn.classList.add('hidden');
            retryCount = 0;
            recentQuestions.length = 0;
            
            // Fade in setup view
            setTimeout(() => {
                setupView.style.opacity = '1';
            }, 50);
            
            updateNetworkStatus();
        }, 300);
    }
    
    // Modal functions
    function closeModal() {
        summaryModal.classList.remove('active');
        setTimeout(() => {
            summaryModal.classList.add('hidden');
        }, 300);
    }
    
    function showInterviewSummary() {
        if (!sessionId) {
            alert('No active interview session.');
            return;
        }
        
        // Show loading state
        setButtonLoading(summaryBtn, true);
        updateStatus('Generating interview summary...');
        
        // Fetch summary from server
        fetch('/interview-summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate summary');
            }
            return response.json();
        })
        .then(data => {
            // Populate the modal with summary data
            populateSummaryModal(data.summary, data.performance_score);
            
            // Show the modal
            summaryModal.classList.remove('hidden');
            setTimeout(() => {
                summaryModal.classList.add('active');
            }, 10);
            
            // Reset loading state
            setButtonLoading(summaryBtn, false);
            updateStatus('Interview completed successfully!');
        })
        .catch(error => {
            console.error('Error fetching summary:', error);
            alert(`Failed to generate summary: ${error.message}`);
            setButtonLoading(summaryBtn, false);
        });
    }
    
    function populateSummaryModal(summary, performanceScore) {
        // Overall rating
        const overallRating = document.getElementById('overall-rating');
        overallRating.textContent = summary.overall_rating;
        overallRating.className = 'rating-display';
        
        // Add appropriate class based on rating
        const ratingLower = summary.overall_rating.toLowerCase();
        if (ratingLower.includes('excellent')) {
            overallRating.classList.add('rating-excellent');
        } else if (ratingLower.includes('good')) {
            overallRating.classList.add('rating-good');
        } else if (ratingLower.includes('average') && !ratingLower.includes('below')) {
            overallRating.classList.add('rating-average');
        } else if (ratingLower.includes('below')) {
            overallRating.classList.add('rating-below-average');
        } else if (ratingLower.includes('poor')) {
            overallRating.classList.add('rating-poor');
        }
        
        // Performance score
        const scoreDisplay = document.getElementById('performance-score-display');
        const score = Math.round(performanceScore * 100);
        scoreDisplay.textContent = `${score}%`;
        
        // Set background color based on score
        let scoreColor;
        if (score >= 90) {
            scoreColor = 'var(--rating-excellent)';
        } else if (score >= 75) {
            scoreColor = 'var(--rating-good)';
        } else if (score >= 60) {
            scoreColor = 'var(--rating-average)';
        } else if (score >= 40) {
            scoreColor = 'var(--rating-below-average)';
        } else {
            scoreColor = 'var(--rating-poor)';
        }
        scoreDisplay.style.backgroundColor = scoreColor;
        
        // Strengths
        const strengthsList = document.getElementById('strengths-list');
        strengthsList.innerHTML = '';
        summary.strengths.forEach(strength => {
            const li = document.createElement('li');
            li.textContent = strength;
            strengthsList.appendChild(li);
        });
        
        // Areas for improvement
        const improvementList = document.getElementById('improvement-list');
        improvementList.innerHTML = '';
        summary.areas_for_improvement.forEach(area => {
            const li = document.createElement('li');
            li.textContent = area;
            improvementList.appendChild(li);
        });
        
        // Skill assessment
        const skillAssessment = document.getElementById('skill-assessment');
        skillAssessment.innerHTML = '';
        
        if (summary.skill_assessment && Object.keys(summary.skill_assessment).length > 0) {
            Object.entries(summary.skill_assessment).forEach(([skill, rating]) => {
                const skillItem = document.createElement('div');
                skillItem.className = 'skill-item';
                
                const skillName = document.createElement('div');
                skillName.className = 'skill-name';
                skillName.textContent = skill;
                
                const skillRating = document.createElement('div');
                skillRating.className = 'skill-rating';
                
                const ratingBar = document.createElement('div');
                ratingBar.className = 'skill-rating-bar';
                
                // Convert rating to percentage if it's a number
                let ratingValue;
                if (typeof rating === 'number') {
                    ratingValue = rating * 100;
                } else if (typeof rating === 'string') {
                    // Handle string ratings like 'Excellent', 'Good', etc.
                    const ratingMap = {
                        'excellent': 90,
                        'good': 75,
                        'average': 60,
                        'below average': 40,
                        'poor': 20
                    };
                    
                    const ratingLower = rating.toLowerCase();
                    ratingValue = 50; // Default
                    
                    for (const [key, value] of Object.entries(ratingMap)) {
                        if (ratingLower.includes(key)) {
                            ratingValue = value;
                            break;
                        }
                    }
                }
                
                // Set width and color based on rating
                ratingBar.style.width = `${ratingValue}%`;
                
                if (ratingValue >= 90) {
                    ratingBar.style.backgroundColor = 'var(--rating-excellent)';
                } else if (ratingValue >= 75) {
                    ratingBar.style.backgroundColor = 'var(--rating-good)';
                } else if (ratingValue >= 60) {
                    ratingBar.style.backgroundColor = 'var(--rating-average)';
                } else if (ratingValue >= 40) {
                    ratingBar.style.backgroundColor = 'var(--rating-below-average)';
                } else {
                    ratingBar.style.backgroundColor = 'var(--rating-poor)';
                }
                
                skillRating.appendChild(ratingBar);
                skillItem.appendChild(skillName);
                skillItem.appendChild(skillRating);
                skillAssessment.appendChild(skillItem);
            });
        } else {
            const noSkills = document.createElement('p');
            noSkills.textContent = 'No specific skills were assessed during this interview.';
            skillAssessment.appendChild(noSkills);
        }
        
        // Recommendations
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = '';
        summary.recommendations.forEach(recommendation => {
            const li = document.createElement('li');
            li.textContent = recommendation;
            recommendationsList.appendChild(li);
        });
    }

    function updateStatus(text) {
        // Animate status text change
        statusText.style.opacity = '0';
        setTimeout(() => {
            statusText.textContent = text;
            statusText.style.opacity = '1';
        }, 200);
    }
    
    // Function to set button loading state
    function setButtonLoading(button, isLoading) {
        if (isLoading) {
            const originalText = button.innerHTML;
            button.setAttribute('data-original-text', originalText);
            button.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...';
            button.disabled = true;
        } else {
            const originalText = button.getAttribute('data-original-text');
            if (originalText) {
                button.innerHTML = originalText;
            }
            button.disabled = false;
        }
    }
});
