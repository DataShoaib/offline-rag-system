import React from 'react';
import { format } from 'date-fns';
import DOMPurify from 'dompurify'; // For sanitizing HTML from system messages

// Assume Message.css is imported or styles are defined in ChatPage.css
// import './Message.css'; 

function Message({ message, onCitationClick }) {
    const isUser = message.sender === 'user';
    const isAi = message.sender === 'ai';
    const isSystem = message.sender === 'system';

    // Determine message styling class
    const messageClass = isUser ? 'user-message' : isAi ? 'ai-message' : 'system-message';

    const handleCitationClick = (file_id, citationText) => {
        if (!file_id) {
            alert(`Source not available for "${citationText || 'this citation'}". File may have been deleted or not uploaded correctly.`);
            return;
        }
        onCitationClick(file_id);
    };

    // Function to render message text with inline clickable citation numbers
    const renderMessageContent = (text, citations) => {
        // If it's a system message, sanitize and render raw HTML (e.g., for <strong> tags)
        if (isSystem) {
            return <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(text) }} />;
        }
        
        // For user messages, just return the text
        if (isUser || !text || !citations || citations.length === 0) {
            return text || '';
        }

        // For AI messages, process for inline citations
        const parts = [];
        let lastIndex = 0;

        // Regex to find citation markers like [1], [2], etc.
        const regex = /\[(\d+)\]/g; 
        let match;

        while ((match = regex.exec(text)) !== null) {
            const citationNumber = parseInt(match[1], 10); // Extract number
            const citationMarker = match[0]; // e.g., "[1]"

            // Add text before the current citation marker
            if (match.index > lastIndex) {
                parts.push(<React.Fragment key={`text-${lastIndex}`}>{text.substring(lastIndex, match.index)}</React.Fragment>);
            }

            // Find the corresponding citation object
            const citation = citations.find(c => c.text && c.text.startsWith(`[${citationNumber}]`));

            if (citation && citation.file_id) {
                // Render clickable citation if file_id exists
                parts.push(
                    <span
                        key={`citation-inline-${match.index}`}
                        className="citation-link clickable-citation"
                        onClick={() => handleCitationClick(citation.file_id, citation.text)}
                        title={`View Source: ${citation.text}`}
                        style={{ cursor: 'pointer', color: '#007bff', textDecoration: 'underline' }}
                    >
                        {citationMarker}
                    </span>
                );
            } else {
                // Render non-clickable citation if file_id is missing or invalid
                parts.push(
                    <span
                        key={`citation-inline-plain-${match.index}`}
                        className="citation-link"
                        title="Source file unavailable or not uploaded correctly"
                        style={{ cursor: 'default', color: '#6c757d', textDecoration: 'none' }}
                    >
                        {citationMarker}
                    </span>
                );
            }

            lastIndex = match.index + citationMarker.length;
        }

        // Add any remaining text after the last citation
        if (lastIndex < text.length) {
            parts.push(<React.Fragment key={`text-end`}>{text.substring(lastIndex)}</React.Fragment>);
        }

        return <>{parts}</>;
    };

    // Only show citation list for AI messages with at least one valid citation (file_id present)
    const hasValidCitations = isAi && message.citations && message.citations.some(c => c.file_id);
    
    // Format timestamp
    const timestamp = message.timestamp ? format(new Date(message.timestamp), 'p') : 'Unknown time';

    return (
        <div className={`message ${messageClass} ${message.isError ? 'error-message' : ''}`}>
            <div className="message-bubble">
                {renderMessageContent(message.text, message.citations)}

                {/* Only show citation list for AI messages with valid citations */}
                {hasValidCitations && (
                    <div className="citation-list-container">
                        <strong>Sources:</strong>
                        <ul>
                            {message.citations
                                .filter(c => c.file_id) // Only show citations with valid file_id in the list
                                .map((c, i) => (
                                    <li key={i}>
                                        <span
                                            className="citation-link clickable-citation-list-item"
                                            onClick={() => handleCitationClick(c.file_id, c.text)}
                                            title={`View Source: ${c.text}`}
                                            style={{
                                                cursor: 'pointer',
                                                color: '#007bff',
                                                textDecoration: 'underline'
                                            }}
                                        >
                                            {c.text || `Source ${i + 1}`} {/* Fallback text */}
                                        </span>
                                    </li>
                                ))}
                        </ul>
                    </div>
                )}
            </div>

            <span className="message-timestamp">{timestamp}</span>
        </div>
    );
}

export default Message;