// Chat functionality
const chatButton = document.getElementById('chatButton');
const chatBox = document.getElementById('chatBox');
const closeButton = document.getElementById('closeButton');
const sendButton = document.getElementById('sendButton');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');
const aiNotice = document.getElementById('aiNotice');
const dismissNotice = document.getElementById('dismissNotice');
const promptButtons = document.querySelectorAll('.prompt-button');

// Toggle chat box
chatButton.addEventListener('click', () => {
    chatBox.classList.remove('hidden');
    chatButton.style.display = 'none';
    messageInput.focus();
});

closeButton.addEventListener('click', () => {
    chatBox.classList.add('hidden');
    chatButton.style.display = 'flex';
});

// Dismiss AI notice
dismissNotice.addEventListener('click', () => {
    aiNotice.classList.add('hidden');
});

// Handle prompt buttons
promptButtons.forEach(button => {
    button.addEventListener('click', () => {
        const prompt = button.getAttribute('data-prompt');
        messageInput.value = prompt;
        sendMessage();
    });
});

// Send message on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send message function
async function sendMessage() {
    const question = messageInput.value.trim();
    
    if (!question) {
        return;
    }
    
    // Clear input
    messageInput.value = '';
    messageInput.disabled = true;
    sendButton.disabled = true;
    
    // Remove suggested prompts after first message
    const suggestedPrompts = document.querySelector('.suggested-prompts');
    if (suggestedPrompts) {
        suggestedPrompts.remove();
    }
    
    // Add user message
    addMessage(question, 'user');
    
    // Add loading indicator
    const loadingId = addLoadingMessage();
    
    try {
        // Call API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        if (data.success) {
            // Add assistant response
            addMessage(data.answer, 'assistant');
        } else {
            // Show error
            addMessage(`Error: ${data.error || 'Failed to get response'}`, 'assistant');
        }
    } catch (error) {
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        // Show error message
        addMessage(`Error: ${error.message || 'Network error. Please try again.'}`, 'assistant');
    } finally {
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'message-icon';
    
    if (type === 'assistant') {
        iconDiv.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="currentColor"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        `;
    } else {
        iconDiv.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        `;
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const p = document.createElement('p');
    // Convert newlines to <br> and preserve formatting
    p.innerHTML = text.replace(/\n/g, '<br>');
    
    contentDiv.appendChild(p);
    messageDiv.appendChild(iconDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// Add loading message
function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = 'loading-message';
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'message-icon';
    iconDiv.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="currentColor"/>
            <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    `;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = '<span></span><span></span><span></span>';
    
    contentDiv.appendChild(loadingDiv);
    messageDiv.appendChild(iconDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'loading-message';
}

// Remove loading message
function removeLoadingMessage(id) {
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// Send button click handler
sendButton.addEventListener('click', sendMessage);

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (data.status === 'ok') {
            console.log('API is ready');
        } else {
            console.warn('API health check failed:', data.message);
        }
    } catch (error) {
        console.error('Failed to check API health:', error);
    }
});

