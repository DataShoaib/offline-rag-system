import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/LoginPage.css'; // This is where the styles are imported
import ntroLogo from '../assets/ntro_logo.png'; // Make sure you have this logo in assets

function LoginPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [loginMessage, setLoginMessage] = useState('');
    const [isLoginSuccess, setIsLoginSuccess] = useState(false);
    const navigate = useNavigate();
    const { login, isAuthenticated } = useAuth();

    useEffect(() => {
        if (isAuthenticated) {
            navigate('/chat');
        }
    }, [isAuthenticated, navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setLoginMessage('');
        setIsLoginSuccess(false);

        try {
            await login(username, password);
            setIsLoginSuccess(true);
            setLoginMessage('Login Successful!');
            setTimeout(() => {
                navigate('/chat');
            }, 1500); // Redirect after a short delay
        } catch (error) {
            setIsLoginSuccess(false);
            setLoginMessage(error.response?.data?.detail || 'Login failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-circle-container">
                <div className="login-circle">
                    <div className="login-header">
                        <img src={ntroLogo} alt="NTRO Logo" className="ntro-logo" />
                        <h1>NTRO</h1>
                        <p>NTRO Intelligence Agency</p>
                    </div>
                    <form onSubmit={handleSubmit} className="login-form">
                        <div className="input-group">
                            {/* htmlFor and id attributes link the label to the input, making the label "clickable" */}
                            <label htmlFor="username">USERNAME</label>
                            <input
                                type="text"
                                id="username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                className="glowing-input"
                            />
                        </div>
                        <div className="input-group">
                            {/* htmlFor and id attributes link the label to the input, making the label "clickable" */}
                            <label htmlFor="password">PASSWORD</label>
                            <input
                                type="password"
                                id="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                className="glowing-input"
                            />
                        </div>
                        <button type="submit" className="access-button glowing-button" disabled={loading}>
                            {loading ? 'ACCESSING...' : 'ACCESS'}
                        </button>
                    </form>
                    {loading && (
                        <div className="verifying-section">
                            <div className="loading-bar">
                                <div className="loading-progress"></div>
                            </div>
                            <p>Verifying Credentials...</p>
                        </div>
                    )}
                    {loginMessage && (
                        <div className={`login-message ${isLoginSuccess ? 'success' : 'error'}`}>
                            {isLoginSuccess && <span className="tick-mark">✔</span>}
                            {loginMessage}
                        </div>
                    )}
                </div>
            </div>
            <div className="animated-background"></div> {/* This will hold the background animation */}
        </div>
    );
}

export default LoginPage;