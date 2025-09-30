import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/ChatPage.css';
// import ntroLogo from '../assets/ntro_logo.png'; // No longer needed here, moved to Sidebar
import Message from './Message';
import Sidebar from './Sidebar'; // Import the Sidebar component

function ChatPage() {
    const { user, logout, checkAuth } = useAuth();
    const navigate = useNavigate();
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const messagesEndRef = useRef(null);
    const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const fetchHistory = useCallback(async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                logout();
                navigate('/login');
                return;
            }
            const response = await axios.get(`${API_BASE_URL}/memory`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setHistory(response.data.history);
        } catch (error) {
            console.error("Error fetching history:", error);
            if (error.response?.status === 401 || error.response?.status === 403) {
                logout();
                navigate('/login');
            }
        }
    }, [logout, navigate, API_BASE_URL]);

    useEffect(() => {
        checkAuth();
        fetchHistory();
    }, [checkAuth, fetchHistory]);

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!inputMessage.trim() || loading) return;

        const userMessage = { text: inputMessage, sender: 'user', timestamp: new Date().toISOString() };
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setInputMessage('');
        setLoading(true);

        try {
            const token = localStorage.getItem('access_token');
            const response = await axios.post(`${API_BASE_URL}/query`, { question: inputMessage }, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const aiMessage = {
                text: response.data.answer,
                sender: 'ai',
                citations: response.data.citations,
                timestamp: new Date().toISOString()
            };
            setMessages((prevMessages) => [...prevMessages, aiMessage]);
            fetchHistory(); // Refresh history after a new query
        } catch (error) {
            console.error("Error querying RAG:", error);
            const errorMessage = {
                text: `Error: ${error.response?.data?.detail || 'Could not get a response.'}`,
                sender: 'ai',
                isError: true,
                timestamp: new Date().toISOString()
            };
            setMessages((prevMessages) => [...prevMessages, errorMessage]);
            if (error.response?.status === 401 || error.response?.status === 403) {
                logout();
                navigate('/login');
            }
        } finally {
            setLoading(false);
        }
    };

    // handleLogout is now in Sidebar component

    const loadHistoryItem = (item) => {
        const loadedMessages = [
            { text: item.query, sender: 'user', timestamp: item.timestamp },
            { text: item.answer, sender: 'ai', citations: item.citations.map(c => ({text: c, file_id: c.match(/\[(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})\]/)?.[1] || ''})), timestamp: item.timestamp }
        ];
        setMessages(loadedMessages);
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const token = localStorage.getItem('access_token');
            await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data',
                },
            });
            alert('File uploaded and ingested successfully!');
        } catch (error) {
            console.error("Error uploading file:", error);
            alert(`File upload failed: ${error.response?.data?.detail || error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const viewCitationSource = async (file_id) => {
        try {
            const token = localStorage.getItem('access_token');
            const response = await axios.get(`${API_BASE_URL}/view_source/${file_id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            alert(`Source Content:\n\n${response.data.content.substring(0, 500)}...`);
        } catch (error) {
            console.error("Error viewing source:", error);
            alert(`Failed to view source: ${error.response?.data?.detail || error.message}`);
        }
    };

    return (
        <div className="chat-container">
            <Sidebar
                history={history}
                loadHistoryItem={loadHistoryItem}
                handleFileUpload={handleFileUpload}
                loading={loading}
            />

            <div className="chat-main">
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="welcome-message">
                            <h2>Welcome, {user?.username}!</h2>
                            <p>Ask me anything about the ingested documents.</p>
                        </div>
                    )}
                    {messages.map((msg, index) => (
                        <Message
                            key={index}
                            message={msg}
                            onCitationClick={viewCitationSource}
                        />
                    ))}
                    {loading && (
                        <div className="message ai-message loading-message">
                            <span>Thinking...</span>
                            <div className="loading-dots">
                                <span>.</span><span>.</span><span>.</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
                <form onSubmit={handleSendMessage} className="chat-input-form">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Ask NTRO AI..."
                        disabled={loading}
                    />
                    <button type="submit" disabled={loading}>
                        <span className="send-icon">➤</span>
                    </button>
                </form>
                <p className="disclaimer">
                    AI may make mistakes. Consider checking important information.
                </p>
            </div>
        </div>
    );
}

export default ChatPage;