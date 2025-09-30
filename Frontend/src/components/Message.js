import React from 'react';

function Message({ message, onCitationClick }) {
  const isUser = message.sender === 'user';
  const messageClass = isUser ? 'user-message' : 'ai-message';

  // Function to render text with citations as clickable links
  const renderMessageContent = (text, citations) => {
    if (!citations || citations.length === 0) {
      return text;
    }

    let content = text;
    const parts = [];
    let lastIndex = 0;

    citations.forEach((citation, index) => {
      const citationRegex = new RegExp(`\\[${index + 1}\\]`); // Match [1], [2] etc.
      const match = citationRegex.exec(content);

      if (match) {
        // Add text before the citation
        if (match.index > lastIndex) {
          parts.push(content.substring(lastIndex, match.index));
        }

        // Add the clickable citation link
        parts.push(
          <span
            key={`citation-${index}`}
            className="citation-link"
            onClick={() => onCitationClick(citation.file_id)}
          >
            {citation.text}
          </span>
        );
        lastIndex = match.index + match[0].length;
      }
    });

    // Add any remaining text
    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return <>{parts}</>;
  };

  return (
    <div className={`message ${messageClass} ${message.isError ? 'error-message' : ''}`}>
      <div className="message-bubble">
        {renderMessageContent(message.text, message.citations)}
      </div>
      <span className="message-timestamp">
        {new Date(message.timestamp).toLocaleTimeString()}
      </span>
    </div>
  );
}

export default Message;