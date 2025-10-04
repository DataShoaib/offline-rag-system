import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/ChatPage.css'; // Make sure this path is correct
import ntroLogo from '../assets/ntro_logo.png'; // Make sure this path is correct
import Message from './Message'; // Make sure this path is correct
import { format } from 'date-fns';

function ChatPage() {
    const { user, logout, checkAuth } = useAuth();
    const navigate = useNavigate();
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [usersList, setUsersList] = useState([]); // For Admin panel
    const [newUsername, setNewUsername] = useState(''); // For Admin panel
    const [newPassword, setNewPassword] = useState(''); // For Admin panel
    const [currentConversationId, setCurrentConversationId] = useState(null);
    const messagesEndRef = useRef(null);
    const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

    // Scroll to the bottom of the chat area
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Fetch conversation history for the current user
    const fetchHistory = useCallback(async () => {
        const token = localStorage.getItem('access_token');
        if (!token) {
            console.warn("Token missing in fetchHistory");
            return;
        }
        try {
            const response = await axios.get(`${API_BASE_URL}/memory`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            // Ensure history is sorted by timestamp descending for display
            const sortedHistory = (response.data || []).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            setHistory(sortedHistory);
        } catch (error) {
            console.error("Error fetching history:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            }
            setHistory([]);
        }
    }, [API_BASE_URL, logout, navigate]);

    // Fetch list of users for Admin panel
    const fetchUsers = useCallback(async () => {
        if (user?.role !== 'Admin') return; // Only Admin can view users
        const token = localStorage.getItem('access_token');
        if (!token) return;
        try {
            const response = await axios.get(`${API_BASE_URL}/users`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            // Assuming response.data is directly the array of users as per `main.py` update
            setUsersList(Array.isArray(response.data) ? response.data : []);
        } catch (error) {
            console.error("Error fetching users list:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            }
            setUsersList([]);
        }
    }, [user, API_BASE_URL, logout, navigate]);

    // Effects for authentication, history, and users list
    useEffect(() => { checkAuth(); }, [checkAuth]);
    useEffect(() => { 
        if (user) { 
            fetchHistory(); 
            // Only fetch users if the current user is an admin
            if (user.role === 'Admin') {
                fetchUsers(); 
            }
        }
    }, [user, fetchHistory, fetchUsers]);
    
    // Effect to scroll to bottom whenever messages update
    useEffect(scrollToBottom, [messages]);

    // Handle sending a new message to the AI
    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!inputMessage.trim() || loading) return;
        if (!user) return alert("You must be logged in to send messages");

        const userMessage = { text: inputMessage, sender: 'user', timestamp: new Date().toISOString() };
        setMessages(prev => [...prev, userMessage]); // Add user message to chat
        setInputMessage(''); // Clear input
        setLoading(true); // Show loading indicator

        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("Token missing! Please login again.");
            setLoading(false);
            return;
        }

        try {
            const payload = {
                question: inputMessage,
                conversation_id: currentConversationId // Include current conversation ID
            };
            const response = await axios.post(`${API_BASE_URL}/query`, payload, {
                headers: { Authorization: `Bearer ${token}` }
            });

            const aiMessage = {
                text: response.data.answer || "No answer available",
                sender: 'ai',
                citations: response.data.citations || [], // Include citations
                timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, aiMessage]); // Add AI message to chat
            setCurrentConversationId(response.data.conversation_id); // Update conversation ID
            
            // Re-fetch history to show the latest conversation turn
            fetchHistory(); 
        } catch (error) {
            console.error("Error querying RAG:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
                setLoading(false);
                return;
            }
            // Display error message in chat
            setMessages(prev => [...prev, {
                text: `Error: ${error.response?.data?.detail || 'Could not get a response.'}`,
                sender: 'ai', // Still display as AI message but with error styling
                isError: true,
                timestamp: new Date().toISOString()
            }]);
        } finally { 
            setLoading(false); // Hide loading indicator
        }
    };

    // Handle user logout
    const handleLogout = () => { 
        logout(); 
        navigate('/login'); 
    };

    // Load a selected conversation from history into the chat area
    const loadHistoryItem = (conversationEntry) => {
        if (conversationEntry && Array.isArray(conversationEntry.messages)) {
            // Map messages to the format expected by the Message component
            const convMessages = conversationEntry.messages.map(msg => ({
                text: msg.text,
                sender: msg.sender,
                citations: msg.citations || [],
                timestamp: msg.timestamp
            }));
            setMessages(convMessages);
            setCurrentConversationId(conversationEntry.id); // Set current conversation ID
        } else {
            setMessages([]);
            setCurrentConversationId(null);
        }
    };

    // Start a new chat (clear current messages)
    const handleNewChat = () => {
        setMessages([]);
        setInputMessage('');
        setCurrentConversationId(null); // Clear current conversation ID
    };

    // Handle file upload
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setLoading(true);
        // Add a system message indicating upload in progress
        setMessages(prev => [...prev, { text: `Uploading <strong>${file.name}</strong>...`, sender: 'system', timestamp: new Date().toISOString() }]);

        const allowedTypes = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
            "image/png", "image/jpeg", "image/gif",
            "audio/mpeg", "audio/wav"
        ];
        if (!allowedTypes.includes(file.type)) {
            setMessages(prev => [...prev, { text: `Upload failed: File type "${file.type}" not supported. Please upload PDF, DOCX, PNG, JPEG, GIF, MP3, or WAV.`, sender: 'system', isError: true, timestamp: new Date().toISOString() }]);
            setLoading(false);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        const token = localStorage.getItem('access_token');
        if (!token) {
            setMessages(prev => [...prev, { text: "Upload failed: Authentication token missing. Please log in again.", sender: 'system', isError: true, timestamp: new Date().toISOString() }]);
            setLoading(false);
            return;
        }

        try {
            const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'multipart/form-data' },
                // Optional: Show upload progress
                onUploadProgress: (progressEvent) => {
                    const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setMessages(prev => {
                        const msgs = [...prev];
                        // Update the last system message if it's the upload progress for this file
                        if (msgs.length > 0 && msgs[msgs.length - 1].sender === 'system' && msgs[msgs.length - 1].text.startsWith('Uploading')) {
                            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], text: `Uploading <strong>${file.name}</strong>... ${percent}%` };
                        } else {
                            // Or add a new one if it's the first progress update
                            msgs.push({ text: `Uploading <strong>${file.name}</strong>... ${percent}%`, sender: 'system', timestamp: new Date().toISOString() });
                        }
                        return msgs;
                    });
                }
            });
            // Final success message
            setMessages(prev => [...prev, { text: `File <strong>${file.name}</strong> uploaded and ingested successfully! (ID: <strong>${response.data.file_id}</strong>)`, sender: 'system', timestamp: new Date().toISOString() }]);
        } catch (error) {
            console.error("Error uploading file:", error);
            let errorMessage = "Upload failed: An unexpected error occurred.";
            if (error.response?.status === 401) {
                errorMessage = "Upload failed: Session expired. Please log in again.";
                logout(); // Log out if session expired
                navigate('/login');
            } else if (error.response?.data?.detail) {
                errorMessage = `Upload failed: ${error.response.data.detail}`;
            } else {
                errorMessage = `Upload failed: ${error.message}`;
            }
            setMessages(prev => [...prev, { text: errorMessage, sender: 'system', isError: true, timestamp: new Date().toISOString() }]);
        } finally { 
            setLoading(false); 
            // Clear the file input after upload attempt
            e.target.value = null; 
        }
    };

    // Handle viewing the source file for a citation
    const viewCitationSource = async (file_id) => {
        if (!file_id) {
            alert("Source not available: File ID is missing.");
            return;
        }
        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("Token missing! Please login again.");
            return;
        }
        try {
            const response = await axios.get(`${API_BASE_URL}/view/${file_id}`, {
                headers: { Authorization: `Bearer ${token}` },
                responseType: 'blob' // Important for downloading files
            });

            // Extract filename from content-disposition header if available, otherwise construct one
            const contentDisposition = response.headers['content-disposition'];
            let filename = `document-${file_id}`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1];
                }
            } else if (response.data.type) {
                const parts = response.data.type.split('/');
                if (parts.length > 1) filename += `.${parts[1]}`; // Fallback for extension
            }

            // Create a temporary URL and download link
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('target', '_blank'); // Open in new tab
            link.setAttribute('download', filename); // Suggest filename for download
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url); // Clean up the temporary URL
        } catch (error) {
            console.error("Error viewing source:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            } else if (error.response?.status === 404) {
                alert("Source file not found. It may have been deleted or not uploaded correctly. Please upload the file again.");
            } else if (error.response?.status === 500) {
                alert("Server error while accessing source. Check backend logs for details or encryption key validity.");
            } else {
                alert(`Failed to view source: ${error.response?.data?.detail || error.message}`);
            }
        }
    };

    // Handle creating a new user (Admin function)
    const handleCreateUser = async (e) => {
        e.preventDefault();
        if (!newUsername || !newPassword) return alert("Please enter both username and password.");
        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("Token missing! Please login again.");
            return;
        }
        try {
            const payload = { username: newUsername, password: newPassword, role: "User" }; // Default to "User" role
            const response = await axios.post(`${API_BASE_URL}/users`, payload, { headers: { Authorization: `Bearer ${token}` } });
            setNewUsername('');
            setNewPassword('');
            await fetchUsers(); // Refresh user list
            alert(response.data.message || "User created successfully!");
        } catch (error) {
            console.error("Error creating user:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            } else {
                const errorMsg = error.response?.data?.detail || error.response?.data?.message || error.message;
                alert(`Create user failed: ${errorMsg}`);
            }
        }
    };

    // Handle deleting a user (Admin function)
    const handleDeleteUser = async (usernameToDelete) => {
        if (!window.confirm(`Are you sure you want to delete user "${usernameToDelete}"? This action cannot be undone.`)) return;
        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("Token missing! Please login again.");
            return;
        }
        try {
            await axios.delete(`${API_BASE_URL}/users/${usernameToDelete}`, { headers: { Authorization: `Bearer ${token}` } });
            await fetchUsers(); // Refresh user list
            alert("User deleted successfully!");
        } catch (error) {
            console.error("Error deleting user:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            } else {
                alert(`Delete failed: ${error.response?.data?.detail || error.message}`);
            }
        }
    };

    // Format history date for display
    const formatHistoryDate = (timestamp) => {
        const date = new Date(timestamp);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(today.getDate() - 1);

        if (date.toDateString() === today.toDateString()) {
            return "Today";
        } else if (date.toDateString() === yesterday.toDateString()) {
            return "Yesterday";
        } else {
            return format(date, 'MMM d, yyyy'); // e.g., "Jan 1, 2023"
        }
    };

    return (
        <div className="ntro-chat-container">
            <div className="background-animation-overlay"></div> {/* Optional: for background effects */}
            
            <div className={`ntro-main-panel ${user?.role === 'Admin' ? 'admin-mode' : ''}`}>
                {/* Left Panel: Logo, New Chat, History, Upload, Admin Panel, User Info, Logout */}
                <div className="ntro-left-panel">
                    <div className="ntro-agency-header">
                        <img src={ntroLogo} alt="NTRO Logo" className="ntro-header-logo" />
                        <div className="agency-title">
                            <h1>NTRO</h1>
                            <p>Intelligence Agency</p>
                        </div>
                    </div>

                    <div className="ntro-new-chat-section">
                        <button onClick={handleNewChat} className="ntro-new-chat-button">+ New Chat</button>
                    </div>

                    <div className="ntro-history-section">
                        <h3>HISTORY</h3>
                        <ul className="ntro-history-list">
                            {history.length === 0 ? (
                                <li className="ntro-no-history">No past conversations.</li>
                            ) : (
                                history.map((item) => {
                                    // Find the first user message for a display title
                                    const firstUserMessage = (item && item.messages && Array.isArray(item.messages))
                                        ? item.messages.find(m => m.sender === 'user')
                                        : null;

                                    const displayTitle = firstUserMessage
                                        ? firstUserMessage.text.substring(0, 50) + (firstUserMessage.text.length > 50 ? '...' : '')
                                        : 'Untitled Conversation';

                                    return (
                                        <li 
                                            key={item.id || item.timestamp} 
                                            onClick={() => loadHistoryItem(item)} 
                                            className={`ntro-history-item ${currentConversationId === item.id ? 'active-conversation' : ''}`}
                                        >
                                            <div className="history-item-text">
                                                {displayTitle}
                                                <span className="history-date"> {formatHistoryDate(item.timestamp)}</span>
                                            </div>
                                        </li>
                                    );
                                })
                            )}
                        </ul>
                    </div>

                    <div className="ntro-upload-section">
                        <label htmlFor="file-upload" className="ntro-upload-button">Upload Document</label>
                        <input id="file-upload" type="file" onChange={handleFileUpload} disabled={loading} style={{ display: 'none' }} />
                    </div>

                    {/* Admin Panel (conditionally rendered for Admin users) */}
                    {user?.role === 'Admin' && (
                        <div className="ntro-admin-panel">
                            <h3>Admin Panel</h3>
                            <h4>Create New User</h4>
                            <form onSubmit={handleCreateUser} className="ntro-create-user-form">
                                <input type="text" placeholder="Username" value={newUsername} onChange={(e) => setNewUsername(e.target.value)} required />
                                <input type="password" placeholder="Password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required />
                                <button type="submit">Create User</button>
                            </form>
                            <h4>Existing Users</h4>
                            <ul className="ntro-users-list">
                                {usersList.length === 0 ? (
                                    <li>No users found.</li>
                                ) : (
                                    usersList.map(u => (
                                        <li key={u.username}>
                                            {u.username} ({u.role || "User"})
                                            {/* Prevent admin from deleting themselves or the default admin */}
                                            {u.username !== user.username && u.username !== "admin" && (
                                                <button onClick={() => handleDeleteUser(u.username)} className="delete-user-button">Delete</button>
                                            )}
                                        </li>
                                    ))
                                )}
                            </ul>
                        </div>
                    )}

                    <div className="ntro-user-controls">
                        <div className="ntro-user-info">
                            <span className="user-icon">👤</span> {/* User icon */}
                            <span>{user?.username} ({user?.role})</span>
                        </div>
                        <button onClick={handleLogout} className="ntro-logout-button">Logout</button>
                    </div>
                </div>

                {/* Right Panel: Chat Area and Input */}
                <div className="ntro-right-panel">
                    <div className="ntro-chat-area">
                        {messages.length === 0 && (
                            <div className="ntro-welcome-message">
                                <h2>Welcome, {user?.username || "Guest"}!</h2>
                                <p>Ask me anything about the ingested documents.</p>
                                <p>You can upload PDFs, DOCX, images (JPG, PNG, GIF), and audio files (MP3, WAV).</p>
                            </div>
                        )}

                        {messages.map((msg, index) => (
                            <Message 
                                key={index} // Consider a more stable key for production, like message ID if available
                                message={msg} 
                                onCitationClick={viewCitationSource} 
                            />
                        ))}

                        {loading && (
                            <div className="message ai-message ntro-loading-message">
                                <span>Processing...</span>
                                <div className="loading-dots"><span>.</span><span>.</span><span>.</span></div>
                            </div>
                        )}
                        <div ref={messagesEndRef} /> {/* For auto-scrolling */}
                    </div>

                    <form onSubmit={handleSendMessage} className="ntro-chat-input-form">
                        <input 
                            type="text" 
                            value={inputMessage} 
                            onChange={(e) => setInputMessage(e.target.value)}
                            placeholder="Ask NTRO AI..." 
                            disabled={loading || !user} // Disable if loading or not logged in
                        />
                        <button type="submit" disabled={loading || !user}><span className="ntro-send-icon">➤</span></button>
                    </form>

                    <p className="ntro-disclaimer">
                        AI may make mistakes. Consider checking important information.
                    </p>
                </div>
            </div>
        </div>
    );
}

export default ChatPage;