import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/ChatPage.css';
import ntroLogo from '../assets/ntro_logo.png';
import Message from './Message';
import { format } from 'date-fns';

function ChatPage() {
    const { user, logout, checkAuth } = useAuth();
    const navigate = useNavigate();
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [usersList, setUsersList] = useState([]);
    const [newUsername, setNewUsername] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [currentConversationId, setCurrentConversationId] = useState(null);
    const [citationPreview, setCitationPreview] = useState(null);
    const [previewContent, setPreviewContent] = useState(null);
    const [previewError, setPreviewError] = useState(null);
    const messagesEndRef = useRef(null);
    const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

    const logStyle = 'background: #fdbb2d; color: black; padding: 2px 4px; border-radius: 2px;';

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const fetchHistory = useCallback(async () => {
        const token = localStorage.getItem('access_token');
        if (!token) {
            console.warn("fetchHistory: Token missing");
            return;
        }
        try {
            const response = await axios.get(`${API_BASE_URL}/memory`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const sortedHistory = (response.data || []).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            setHistory(sortedHistory);
            console.log(`fetchHistory: Fetched ${sortedHistory.length} conversations`);
        } catch (error) {
            console.error("fetchHistory: Error fetching history:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            }
            setHistory([]);
        }
    }, [API_BASE_URL, logout, navigate]);

    const fetchUsers = useCallback(async () => {
        if (user?.role !== 'Admin') return;
        const token = localStorage.getItem('access_token');
        if (!token) return;
        try {
            const response = await axios.get(`${API_BASE_URL}/users`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setUsersList(Array.isArray(response.data) ? response.data : []);
            console.log(`fetchUsers: Fetched ${response.data.length} users`);
        } catch (error) {
            console.error("fetchUsers: Error fetching users list:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            }
            setUsersList([]);
        }
    }, [user, API_BASE_URL, logout, navigate]);

    useEffect(() => { checkAuth(); }, [checkAuth]);
    useEffect(() => {
        if (user) {
            fetchHistory();
            if (user.role === 'Admin') {
                fetchUsers();
            }
        }
    }, [user, fetchHistory, fetchUsers]);

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!inputMessage.trim() || loading) return;
        if (!user) return alert("You must be logged in to send messages");

        const userMessage = { text: inputMessage, sender: 'user', timestamp: new Date().toISOString() };
        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setLoading(true);

        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("handleSendMessage: Token missing! Please login again.");
            setLoading(false);
            return;
        }

        try {
            const payload = {
                question: inputMessage,
                conversation_id: currentConversationId
            };
            const response = await axios.post(`${API_BASE_URL}/query`, payload, {
                headers: { Authorization: `Bearer ${token}` }
            });

            console.log("handleSendMessage: Query response citations:", response.data.citations);

            const aiMessage = {
                text: response.data.answer || "No answer available",
                sender: 'ai',
                citations: response.data.citations || [],
                timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, aiMessage]);
            setCurrentConversationId(response.data.conversation_id);
            fetchHistory();
        } catch (error) {
            console.error("handleSendMessage: Error querying RAG:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
                setLoading(false);
                return;
            }
            setMessages(prev => [...prev, {
                text: `Error: ${error.response?.data?.detail || 'Could not get a response.'}`,
                sender: 'ai',
                isError: true,
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const loadHistoryItem = (conversationEntry) => {
        if (conversationEntry && Array.isArray(conversationEntry.messages)) {
            const convMessages = conversationEntry.messages.map(msg => ({
                text: msg.text,
                sender: msg.sender,
                citations: msg.citations || [],
                timestamp: msg.timestamp
            }));
            setMessages(convMessages);
            setCurrentConversationId(conversationEntry.id);
            console.log(`loadHistoryItem: Loaded conversation ${conversationEntry.id}`);
        } else {
            setMessages([]);
            setCurrentConversationId(null);
            console.warn("loadHistoryItem: Invalid conversation entry");
        }
    };

    const handleNewChat = () => {
        setMessages([]);
        setInputMessage('');
        setCurrentConversationId(null);
        console.log("handleNewChat: Started new chat");
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setLoading(true);
        setMessages(prev => [...prev, { text: `Uploading <strong>${file.name}</strong>...`, sender: 'system', timestamp: new Date().toISOString() }]);

        const allowedTypes = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
                onUploadProgress: (progressEvent) => {
                    const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setMessages(prev => {
                        const msgs = [...prev];
                        if (msgs.length > 0 && msgs[msgs.length - 1].sender === 'system' && msgs[msgs.length - 1].text.startsWith('Uploading')) {
                            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], text: `Uploading <strong>${file.name}</strong>... ${percent}%` };
                        } else {
                            msgs.push({ text: `Uploading <strong>${file.name}</strong>... ${percent}%`, sender: 'system', timestamp: new Date().toISOString() });
                        }
                        return msgs;
                    });
                }
            });
            setMessages(prev => [...prev, { text: `File <strong>${file.name}</strong> uploaded and ingested successfully! (ID: <strong>${response.data.file_id}</strong>)`, sender: 'system', timestamp: new Date().toISOString() }]);
            console.log(`handleFileUpload: Uploaded file ${file.name}, file_id=${response.data.file_id}`);
        } catch (error) {
            console.error("handleFileUpload: Error uploading file:", error);
            let errorMessage = "Upload failed: An unexpected error occurred.";
            if (error.response?.status === 401) {
                errorMessage = "Upload failed: Session expired. Please log in again.";
                logout();
                navigate('/login');
            } else if (error.response?.data?.detail) {
                errorMessage = `Upload failed: ${error.response.data.detail}`;
            } else {
                errorMessage = `Upload failed: ${error.message}`;
            }
            setMessages(prev => [...prev, { text: errorMessage, sender: 'system', isError: true, timestamp: new Date().toISOString() }]);
        } finally {
            setLoading(false);
            e.target.value = null;
        }
    };

    const viewCitationSource = async (file_id, file_type, page_num = null, start_time = null, original_filename = "document") => {
        if (!file_id) {
            console.error("viewCitationSource: File ID is missing");
            alert("Source not available: File ID is missing.");
            return;
        }

        console.log(`%cviewCitationSource: file_id=${file_id}, file_type=${file_type}, page_num=${page_num}, start_time=${start_time}, original_filename=${original_filename}`, logStyle);
        setCitationPreview({ file_id, file_type, page_num, start_time, original_filename });
        setPreviewContent(null);
        setPreviewError(null);

        const token = localStorage.getItem('access_token');
        if (!token) {
            console.error("viewCitationSource: Authentication token missing");
            setPreviewError("Authentication token missing. Please log in again.");
            return;
        }

        try {
            const url = file_type === 'PDF' && page_num
                ? `${API_BASE_URL}/preview/${file_id}?page_num=${page_num}`
                : `${API_BASE_URL}/preview/${file_id}${start_time ? `?start_time=${start_time}` : ''}`;
            console.log(`%cviewCitationSource: Fetching preview from ${url}`, logStyle);
            const response = await axios.get(url, {
                headers: { Authorization: `Bearer ${token}` },
                responseType: 'blob'
            });

            const contentType = response.headers['content-type'];
            console.log(`%cviewCitationSource: Response received, content-type=${contentType}, size=${response.data.size} bytes`, logStyle);

            if (contentType.includes('image')) {
                const blobUrl = URL.createObjectURL(response.data);
                console.log(`%cviewCitationSource: Using blob URL: ${blobUrl}`, logStyle);
                setPreviewContent(null); // Reset to trigger re-render
                setTimeout(() => setPreviewContent({ type: 'image', src: blobUrl }), 0);
                // Log data URL for debugging
                const reader = new FileReader();
                reader.onloadend = () => {
                    if (reader.result) {
                        console.log(`%cviewCitationSource: Data URL generated, length=${reader.result.length}`, logStyle);
                    } else {
                        console.error("viewCitationSource: FileReader failed to generate data URL");
                    }
                };
                reader.onerror = () => {
                    console.error("viewCitationSource: FileReader error");
                };
                reader.readAsDataURL(response.data);
            } else if (contentType.includes('audio')) {
                const blobUrl = URL.createObjectURL(response.data);
                console.log(`%cviewCitationSource: Audio blob URL generated: ${blobUrl}`, logStyle);
                setPreviewContent({ type: 'audio', src: `${blobUrl}#t=${start_time || 0}` });
            } else if (contentType.includes('text')) {
                const text = await response.data.text();
                console.log(`%cviewCitationSource: Text content received, length=${text.length}`, logStyle);
                setPreviewContent({ type: 'text', content: text });
            } else {
                console.error(`viewCitationSource: Unsupported content type: ${contentType}`);
                setPreviewError("Unsupported preview content type.");
            }
        } catch (error) {
            console.error("viewCitationSource: Error fetching preview:", error);
            if (error.response?.status === 401) {
                setPreviewError("Session expired. Please log in again.");
                logout();
                navigate('/login');
            } else {
                setPreviewError(`Failed to load preview: ${error.response?.data?.detail || error.message}`);
            }
        }
    };

    const closeCitationPreview = () => {
        if (previewContent?.type === 'image' && previewContent.src.startsWith('blob:')) {
            URL.revokeObjectURL(previewContent.src);
            console.log("closeCitationPreview: Revoked blob URL");
        }
        setCitationPreview(null);
        setPreviewContent(null);
        setPreviewError(null);
        console.log("closeCitationPreview: Modal closed");
    };

    const handleCreateUser = async (e) => {
        e.preventDefault();
        if (!newUsername || !newPassword) return alert("Please enter both username and password.");
        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("handleCreateUser: Token missing! Please login again.");
            return;
        }
        try {
            const payload = { username: newUsername, password: newPassword, role: "User" };
            const response = await axios.post(`${API_BASE_URL}/users`, payload, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setNewUsername('');
            setNewPassword('');
            await fetchUsers();
            alert(response.data.message || "User created successfully!");
            console.log(`handleCreateUser: Created user ${newUsername}`);
        } catch (error) {
            console.error("handleCreateUser: Error creating user:", error);
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

    const handleDeleteUser = async (usernameToDelete) => {
        if (!window.confirm(`Are you sure you want to delete user "${usernameToDelete}"? This action cannot be undone.`)) return;
        const token = localStorage.getItem('access_token');
        if (!token) {
            alert("handleDeleteUser: Token missing! Please login again.");
            return;
        }
        try {
            await axios.delete(`${API_BASE_URL}/users/${usernameToDelete}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            await fetchUsers();
            alert("User deleted successfully!");
            console.log(`handleDeleteUser: Deleted user ${usernameToDelete}`);
        } catch (error) {
            console.error("handleDeleteUser: Error deleting user:", error);
            if (error.response?.status === 401) {
                alert("Session expired. Please log in again.");
                logout();
                navigate('/login');
            } else {
                alert(`Delete failed: ${error.response?.data?.detail || error.message}`);
            }
        }
    };

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
            return format(date, 'MMM d, yyyy');
        }
    };

    const testImageDisplay = () => {
        if (previewContent?.type === 'image' && previewContent.src) {
            window.open(previewContent.src, '_blank');
            console.log(`%ctestImageDisplay: Opened image in new tab: ${previewContent.src}`, logStyle);
        } else {
            alert("No image preview available to test.");
            console.warn("testImageDisplay: No image preview available");
        }
    };

    return (
        <div className="ntro-chat-container">
            <div className="background-animation-overlay"></div>

            <div className={`ntro-main-panel ${user?.role === 'Admin' ? 'admin-mode' : ''}`}>
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
                            <span className="user-icon">👤</span>
                            <span>{user?.username} ({user?.role})</span>
                        </div>
                        <button onClick={handleLogout} className="ntro-logout-button">Logout</button>
                    </div>
                </div>

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
                                key={index}
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
                        <div ref={messagesEndRef} />
                    </div>

                    <form onSubmit={handleSendMessage} className="ntro-chat-input-form">
                        <input
                            type="text"
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            placeholder="Ask NTRO AI..."
                            disabled={loading || !user}
                        />
                        <button type="submit" disabled={loading || !user}><span className="ntro-send-icon">➤</span></button>
                    </form>

                    <p className="ntro-disclaimer">
                        AI may make mistakes. Consider checking important information.
                    </p>
                </div>
            </div>

            {citationPreview && (
                <div className="citation-preview-modal-overlay" onClick={closeCitationPreview}>
                    <div className="citation-preview-modal-content" onClick={e => e.stopPropagation()}>
                        <button className="close-preview-button" onClick={closeCitationPreview}>&times;</button>
                        <h3>
                            Preview: {citationPreview.original_filename}
                            {citationPreview.file_type === 'PDF' && citationPreview.page_num && ` (Page ${citationPreview.page_num})`}
                            {citationPreview.file_type === 'AUDIO' && citationPreview.start_time && ` (Start: ${citationPreview.start_time}s)`}
                        </h3>
                        <div className="preview-content-area">
                            {previewError && (
                                <p className="preview-error">{previewError}</p>
                            )}
                            {!previewContent && !previewError && (
                                <p className="preview-loading">Loading preview...</p>
                            )}
                            {previewContent && previewContent.type === 'image' && (
                                <>
                                    <img
                                        src={previewContent.src}
                                        alt={`Preview for ${citationPreview.original_filename}`}
                                        className="preview-item preview-image"
                                        onError={(e) => {
                                            console.error("Image render error:", e);
                                            setPreviewError("Failed to render image. Please try again.");
                                        }}
                                    />
                                    <p>Image Source: {previewContent.src.startsWith('data:') ? 'Data URL' : 'Blob URL'}</p>
                                    <button className="test-image-button" onClick={testImageDisplay} disabled={!previewContent?.src}>
                                        Test Image in New Tab
                                    </button>
                                </>
                            )}
                            {previewContent && previewContent.type === 'audio' && (
                                <audio
                                    controls
                                    className="preview-item preview-audio"
                                    autoPlay={true}
                                    src={previewContent.src}
                                    onError={(e) => {
                                        console.error("Audio playback error:", e);
                                        setPreviewError("Failed to play audio. Please try again.");
                                    }}
                                >
                                    Your browser does not support the audio element.
                                </audio>
                            )}
                            {previewContent && previewContent.type === 'text' && (
                                <div className="preview-item preview-text">{previewContent.content}</div>
                            )}
                            {citationPreview.file_type !== 'PDF' &&
                             citationPreview.file_type !== 'IMAGE' &&
                             citationPreview.file_type !== 'AUDIO' &&
                             citationPreview.file_type !== 'DOCX' &&
                             citationPreview.file_type !== 'TEXT' && (
                                <p>No direct preview available for this file type. You can still download the original file <a href={`${API_BASE_URL}/view/${citationPreview.file_id}`} target="_blank" rel="noopener noreferrer">here</a>.</p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ChatPage;