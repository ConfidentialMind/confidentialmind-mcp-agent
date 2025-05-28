import hashlib
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation fingerprinting and hash chain operations for stateless conversation tracking."""

    def generate_conversation_id(self, first_message: str) -> str:
        """
        Generate a deterministic conversation ID from the first user message.

        Args:
            first_message: The content of the first user message

        Returns:
            A conversation ID in the format "conv_<hash>"
        """
        # Add timestamp to make conversations unique even with same first message
        timestamp = datetime.now().isoformat()
        content = f"{first_message}|{timestamp}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"conv_{hash_value}"

    def generate_message_hash(self, message: dict, parent_hash: str = "") -> str:
        """
        Generate a hash for a message, chained with the parent hash.

        Args:
            message: Message dict with 'role' and 'content' keys
            parent_hash: Hash of the previous message in the chain

        Returns:
            SHA-256 hash of the message
        """
        # Create deterministic string representation
        content = f"{message.get('role', '')}:{message.get('content', '')}|{parent_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def create_hash_chain(self, messages: List[dict]) -> List[str]:
        """
        Create a hash chain from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            List of message hashes forming a chain
        """
        hashes = []
        parent_hash = ""

        for message in messages:
            message_hash = self.generate_message_hash(message, parent_hash)
            hashes.append(message_hash)
            parent_hash = message_hash

        return hashes

    def find_matching_conversation(self, message_hashes: List[str]) -> Optional[str]:
        """
        Find a conversation ID that matches the given hash chain.

        Note: This is a placeholder - actual implementation will query the database.

        Args:
            message_hashes: List of message hashes to match

        Returns:
            Conversation ID if found, None otherwise
        """
        # This will be implemented in the database operations
        # For now, return None to indicate no match
        return None

    def detect_conversation_branch(
        self, stored_chain: List[str], incoming_chain: List[str]
    ) -> Optional[int]:
        """
        Detect where two conversation chains diverge.

        Args:
            stored_chain: Hash chain from the database
            incoming_chain: Hash chain from incoming messages

        Returns:
            Index of the branch point, or None if chains match
        """
        for i, (stored_hash, incoming_hash) in enumerate(zip(stored_chain, incoming_chain)):
            if stored_hash != incoming_hash:
                logger.info(f"Conversation branch detected at position {i}")
                return i

        # Check if one chain is longer than the other
        if len(stored_chain) != len(incoming_chain):
            logger.info(
                f"Conversation chain length mismatch: stored={len(stored_chain)}, incoming={len(incoming_chain)}"
            )
            return min(len(stored_chain), len(incoming_chain))

        return None  # Chains match completely

    def create_conversation_branch(self, base_conversation_id: str, branch_point: int) -> str:
        """
        Create a new conversation ID for a branched conversation.

        Args:
            base_conversation_id: Original conversation ID
            branch_point: Index where the conversation branched

        Returns:
            New conversation ID for the branch
        """
        # Create branch ID by hashing the base ID with branch point and timestamp
        timestamp = datetime.now().isoformat()
        content = f"{base_conversation_id}|branch_{branch_point}|{timestamp}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"conv_branch_{hash_value}"
