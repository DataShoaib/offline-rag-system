// frontend/src/components/Sidebar.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ntroLogo from '../assets/ntro_logo.png';
import '../styles/ChatPage.css'; // Sidebar styles are in ChatPage.css

function Sidebar({ history, loadHistoryItem, handleFileUpload, loading }) {
    const { user, logout } = useAuth(); // Access user and logout from context
    const navigate = useNavigate();

    const handleLogout = () => {
        logout(); // AuthContext handles navigation after logout
    };

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <img src={ntroLogo} alt="NTRO Logo" className="sidebar-logo" />
                <span>NTRO Intelligence Agency</span>
            </div>
            <div className="sidebar-section">
                <h3>HISTORY</h3>
                <ul className="history-list">
                    {history.length === 0 ? (
                        <li className="no-history">No past conversations.</li>
                    ) : (
                        history.map((item, index) => (
                            <li key={index} onClick={() => loadHistoryItem(item)} className="history-item">
                                {item.query.substring(0, 40)}...
                                <span className="history-timestamp">{new Date(item.timestamp).toLocaleDateString()}</span>
                            </li>
                        ))
                    )}
                </ul>
            </div>
            <div className="sidebar-actions">
                {/* Ensure user?.role exists before checking */}
                {(user && (user.role === 'Admin' || user.role === 'Analyst')) && (
                    <div className="file-upload-section">
                        <label htmlFor="file-upload" className="upload-button">
                            Upload Document
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            onChange={handleFileUpload}
                            disabled={loading}
                            style={{ display: 'none' }}
                        />
                    </div>
                )}
                <div className="user-info">
                    <span className="user-icon">👤</span>
                    <span>{user?.username || 'Guest'} ({user?.role || 'N/A'})</span> {/* Display user info */}
                </div>
                <button onClick={handleLogout} className="logout-button">
                    Logout
                </button>
            </div>
        </div>
    );
}

export default Sidebar;