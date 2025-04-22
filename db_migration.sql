-- Database Migration for Agent Conversation History

-- Create conversation_messages table if it doesn't exist
CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_order INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on session_id for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id ON conversation_messages(session_id);

-- Create index on message_order for efficient ordering within sessions
CREATE INDEX IF NOT EXISTS idx_conversation_messages_order ON conversation_messages(session_id, message_order);

-- Create comment for table
COMMENT ON TABLE conversation_messages IS 'Stores conversation history for agent sessions';

-- Create comments for columns
COMMENT ON COLUMN conversation_messages.id IS 'Primary key';
COMMENT ON COLUMN conversation_messages.session_id IS 'Unique identifier for the conversation session';
COMMENT ON COLUMN conversation_messages.message_order IS 'Order of messages within the session';
COMMENT ON COLUMN conversation_messages.role IS 'Role of the message sender (user or assistant)';
COMMENT ON COLUMN conversation_messages.content IS 'Content of the message';
COMMENT ON COLUMN conversation_messages.timestamp IS 'When the message was created';
