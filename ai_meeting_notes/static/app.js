/**
 * AI Meeting Notes - Web Interface JavaScript
 * 
 * Handles real-time status updates, API communication, and user interactions.
 */

class MeetingNotesApp {
    constructor() {
        this.currentSession = null;
        this.statusUpdateInterval = null;
        this.recordingTimer = null;
        this.recordingStartTime = null;
        
        this.initializeElements();
        this.bindEvents();
        this.checkSystemHealth();
        this.startStatusUpdates();
    }
    
    initializeElements() {
        // Status elements
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.healthStatus = document.getElementById('health-status');
        
        // Control elements
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.testSetupBtn = document.getElementById('test-setup-btn');
        this.retryBtn = document.getElementById('retry-btn');
        
        // Display elements
        this.recordingDuration = document.getElementById('recording-duration');
        this.progressContainer = document.getElementById('progress-container');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        
        // Section elements
        this.setupSection = document.getElementById('setup-section');
        this.transcriptSection = document.getElementById('transcript-section');
        this.notesSection = document.getElementById('notes-section');
        this.errorSection = document.getElementById('error-section');
        
        // Content elements
        this.transcriptContent = document.getElementById('transcript-content');
        this.notesContent = document.getElementById('notes-content');
        this.errorMessage = document.getElementById('error-message');
        
        // Copy buttons
        this.copyTranscriptBtn = document.getElementById('copy-transcript-btn');
        this.copyNotesBtn = document.getElementById('copy-notes-btn');
        this.exportLLMBtn = document.getElementById('export-llm-btn');

        // Toast container
        this.toastContainer = document.getElementById('toast-container');
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.clearBtn.addEventListener('click', () => this.clearSession());
        this.testSetupBtn.addEventListener('click', () => this.testAudioSetup());
        this.retryBtn.addEventListener('click', () => this.retryProcessing());

        this.copyTranscriptBtn.addEventListener('click', () => this.copyToClipboard('transcript'));
        this.copyNotesBtn.addEventListener('click', () => this.copyToClipboard('notes'));
        this.exportLLMBtn.addEventListener('click', () => this.exportForLLM());
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.updateStatus();
            }
        });
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            let statusText = 'System Ready';
            let hasIssues = false;
            
            if (!health.services.blackhole_available) {
                statusText = 'Audio setup required';
                hasIssues = true;
                this.showSetupInstructions();
            } else if (!health.services.ollama_available) {
                statusText = 'Ollama not available';
                hasIssues = true;
            } else if (!health.services.disk_space_ok) {
                statusText = 'Low disk space';
                hasIssues = true;
            }
            
            this.healthStatus.textContent = statusText;
            this.healthStatus.style.color = hasIssues ? 'var(--warning-color)' : 'var(--success-color)';
            
        } catch (error) {
            this.healthStatus.textContent = 'System check failed';
            this.healthStatus.style.color = 'var(--danger-color)';
            console.error('Health check failed:', error);
        }
    }
    
    showSetupInstructions() {
        this.setupSection.style.display = 'block';
        this.startBtn.disabled = true;
    }
    
    hideSetupInstructions() {
        this.setupSection.style.display = 'none';
        this.startBtn.disabled = false;
    }
    
    async testAudioSetup() {
        this.testSetupBtn.disabled = true;
        this.testSetupBtn.textContent = 'Testing...';
        
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            if (health.services.blackhole_available) {
                this.showToast('Audio setup successful!', 'success');
                this.hideSetupInstructions();
            } else {
                this.showToast('BlackHole not detected. Please complete setup steps.', 'error');
            }
        } catch (error) {
            this.showToast('Setup test failed', 'error');
        } finally {
            this.testSetupBtn.disabled = false;
            this.testSetupBtn.textContent = 'Test Audio Setup';
        }
    }
    
    startStatusUpdates() {
        this.updateStatus();
        this.statusUpdateInterval = setInterval(() => {
            this.updateStatus();
        }, 2000); // Update every 2 seconds
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusDisplay(status);
            this.updateProgress(status);
            
            // Check for new results
            if (status.status === 'completed' || status.status === 'error') {
                await this.checkForResults();
            }
            
        } catch (error) {
            console.error('Status update failed:', error);
        }
    }
    
    updateStatusDisplay(status) {
        // Update status indicator
        this.statusDot.className = `status-dot ${status.status}`;
        this.statusText.textContent = this.getStatusText(status);
        
        // Update button states
        const isRecording = status.status === 'recording';
        const isProcessing = ['transcribing', 'generating_notes'].includes(status.status);
        
        this.startBtn.disabled = isRecording || isProcessing;
        this.stopBtn.disabled = !isRecording;
        this.clearBtn.disabled = isRecording || isProcessing;
        
        // Show/hide progress
        if (isProcessing) {
            this.progressContainer.style.display = 'flex';
        } else if (status.status === 'idle' || status.status === 'completed') {
            this.progressContainer.style.display = 'none';
        }
        
        // Update recording duration
        if (isRecording && status.recording_duration) {
            this.updateRecordingDuration(status.recording_duration);
        } else if (!isRecording) {
            this.recordingDuration.textContent = '00:00';
        }
        
        // Handle errors
        if (status.status === 'error' && status.error_message) {
            this.showError(status.error_message);
        } else {
            this.hideError();
        }
    }
    
    getStatusText(status) {
        const statusMap = {
            'idle': 'Ready',
            'recording': `Recording (${this.formatDuration(status.recording_duration || 0)})`,
            'transcribing': 'Transcribing audio...',
            'generating_notes': 'Generating notes...',
            'completed': 'Complete',
            'error': 'Error occurred'
        };
        
        return statusMap[status.status] || status.status;
    }
    
    updateProgress(status) {
        if (status.processing_progress !== undefined) {
            const progress = Math.round(status.processing_progress);
            this.progressFill.style.width = `${progress}%`;
            this.progressText.textContent = `${progress}%`;
        }
    }
    
    updateRecordingDuration(seconds) {
        this.recordingDuration.textContent = this.formatDuration(seconds);
    }
    
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    async startRecording() {
        try {
            this.startBtn.disabled = true;
            this.startBtn.textContent = 'Starting...';
            
            const response = await fetch('/api/start-recording', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start recording');
            }
            
            const result = await response.json();
            this.showToast('Recording started', 'success');
            
        } catch (error) {
            this.showToast(error.message, 'error');
            this.startBtn.disabled = false;
        } finally {
            this.startBtn.textContent = 'Start Recording';
        }
    }
    
    async stopRecording() {
        try {
            this.stopBtn.disabled = true;
            this.stopBtn.textContent = 'Stopping...';
            
            const response = await fetch('/api/stop-recording', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to stop recording');
            }
            
            const result = await response.json();
            this.showToast('Recording stopped, processing started', 'success');
            
        } catch (error) {
            this.showToast(error.message, 'error');
        } finally {
            this.stopBtn.textContent = 'Stop Recording';
        }
    }
    
    async clearSession() {
        if (!confirm('Clear current session? This will delete all data.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/clear', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to clear session');
            }
            
            // Clear UI
            this.transcriptSection.style.display = 'none';
            this.notesSection.style.display = 'none';
            this.hideError();
            
            this.showToast('Session cleared', 'success');
            
        } catch (error) {
            this.showToast(error.message, 'error');
        }
    }
    
    async checkForResults() {
        // Check for transcript
        try {
            const transcriptResponse = await fetch('/api/transcript');
            const transcriptData = await transcriptResponse.json();
            
            if (transcriptData.available && transcriptData.transcript) {
                this.displayTranscript(transcriptData.transcript);
            }
        } catch (error) {
            console.error('Failed to fetch transcript:', error);
        }
        
        // Check for notes
        try {
            const notesResponse = await fetch('/api/notes');
            const notesData = await notesResponse.json();
            
            if (notesData.available && notesData.notes) {
                this.displayNotes(notesData.notes);
            }
        } catch (error) {
            console.error('Failed to fetch notes:', error);
        }
    }
    
    displayTranscript(transcript) {
        if (transcript.full_text) {
            this.transcriptContent.textContent = transcript.full_text;
            this.transcriptSection.style.display = 'block';
        }
    }
    
    displayNotes(notes) {
        const formattedNotes = this.formatNotes(notes);
        this.notesContent.innerHTML = formattedNotes;
        this.notesSection.style.display = 'block';
    }
    
    formatNotes(notes) {
        let html = '';
        
        if (notes.summary) {
            html += `<h3>Summary</h3><p>${notes.summary}</p>`;
        }
        
        if (notes.agenda_items && notes.agenda_items.length > 0) {
            html += '<h3>Agenda Items</h3><ul>';
            notes.agenda_items.forEach(item => {
                html += `<li><strong>${item.title}</strong>`;
                if (item.description) {
                    html += `: ${item.description}`;
                }
                html += '</li>';
            });
            html += '</ul>';
        }
        
        if (notes.discussion_points && notes.discussion_points.length > 0) {
            html += '<h3>Discussion Points</h3>';
            notes.discussion_points.forEach(point => {
                html += `<h4>${point.topic}</h4><ul>`;
                point.key_points.forEach(keyPoint => {
                    html += `<li>${keyPoint}</li>`;
                });
                html += '</ul>';
            });
        }
        
        if (notes.action_items && notes.action_items.length > 0) {
            html += '<h3>Action Items</h3><ul>';
            notes.action_items.forEach(item => {
                html += `<li><strong>${item.task}</strong>`;
                if (item.assignee) {
                    html += ` (Assigned to: ${item.assignee})`;
                }
                if (item.due_date) {
                    html += ` (Due: ${item.due_date})`;
                }
                html += '</li>';
            });
            html += '</ul>';
        }
        
        if (notes.decisions && notes.decisions.length > 0) {
            html += '<h3>Decisions</h3><ul>';
            notes.decisions.forEach(decision => {
                html += `<li><strong>${decision.decision}</strong>`;
                if (decision.rationale) {
                    html += `: ${decision.rationale}`;
                }
                html += '</li>';
            });
            html += '</ul>';
        }
        
        if (notes.participants && notes.participants.length > 0) {
            html += `<h3>Participants</h3><p>${notes.participants.join(', ')}</p>`;
        }
        
        return html;
    }
    
    async exportForLLM() {
        try {
            const response = await fetch('/api/export-prompt');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate export');
            }

            // Copy the export prompt to clipboard
            await navigator.clipboard.writeText(data.prompt);

            // Visual feedback
            const originalText = this.exportLLMBtn.innerHTML;
            this.exportLLMBtn.innerHTML = '✅ Copied!';
            this.exportLLMBtn.style.backgroundColor = 'var(--success-color)';

            setTimeout(() => {
                this.exportLLMBtn.innerHTML = originalText;
                this.exportLLMBtn.style.backgroundColor = '';
            }, 3000);

            this.showToast('Prompt + transcript copied! Paste into ChatGPT or Claude', 'success');

        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Failed to export for LLM: ' + error.message, 'error');
        }
    }

    async copyToClipboard(type) {
        let text = '';
        let button = null;

        if (type === 'transcript') {
            text = this.transcriptContent.textContent;
            button = this.copyTranscriptBtn;
        } else if (type === 'notes') {
            text = this.notesContent.textContent;
            button = this.copyNotesBtn;
        }

        if (!text) {
            this.showToast('Nothing to copy', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);

            // Visual feedback
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.style.backgroundColor = 'var(--success-color)';

            setTimeout(() => {
                button.textContent = originalText;
                button.style.backgroundColor = '';
            }, 2000);

            this.showToast(`${type.charAt(0).toUpperCase() + type.slice(1)} copied to clipboard`, 'success');

        } catch (error) {
            this.showToast('Failed to copy to clipboard', 'error');
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.style.display = 'block';
        this.retryBtn.style.display = 'inline-block';
    }
    
    hideError() {
        this.errorSection.style.display = 'none';
    }
    
    async retryProcessing() {
        // This would trigger a retry of the last failed operation
        // For now, we'll just refresh the status
        this.showToast('Retry functionality not yet implemented', 'warning');
    }
    
    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        this.toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Allow manual dismissal
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MeetingNotesApp();
});