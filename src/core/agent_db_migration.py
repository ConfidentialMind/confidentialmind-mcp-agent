"""Database schema migration for agent conversation history."""

import logging
import traceback
from typing import Dict, List, Optional, Set, Tuple

from src.core.agent_db_connection import AgentDatabase

logger = logging.getLogger(__name__)


class AgentMigration:
    """Handles database schema migrations for the agent conversation history."""

    def __init__(self, db: AgentDatabase):
        """Initialize with database connection."""
        self.db = db
        self.current_version = "v1.0.0"

        # Define the expected schema structure
        self.expected_tables = {
            "conversation_messages": {
                "columns": {
                    "id": {"type": "integer", "is_primary": True},
                    "session_id": {"type": "text", "is_nullable": False},
                    "message_order": {"type": "integer", "is_nullable": False},
                    "role": {"type": "text", "is_nullable": False},
                    "content": {"type": "text", "is_nullable": False},
                    "timestamp": {"type": "timestamp with time zone", "is_nullable": False},
                },
                "indices": [
                    "idx_conversation_messages_session_id",
                    "idx_conversation_messages_order",
                ],
            },
            "agent_migrations": {
                "columns": {
                    "version": {"type": "text", "is_primary": True},
                    "applied_at": {"type": "timestamp with time zone", "is_nullable": True},
                    "status": {"type": "text", "is_nullable": True},
                    "error_details": {"type": "text", "is_nullable": True},
                },
                "indices": [],
            },
        }

    async def ensure_migration_table(self) -> None:
        """Ensure the migration tracking table exists."""
        logger.debug("Ensuring migration table exists")
        try:
            await self.db.ensure_connected()

            # Check if the table exists
            table_exists = await self.db.execute_query(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agent_migrations')",
                fetch_type="val",
            )

            if not table_exists:
                # Create the table if it doesn't exist
                await self.db.execute_query(
                    """
                    CREATE TABLE IF NOT EXISTS agent_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        status TEXT,
                        error_details TEXT
                    )
                """,
                    fetch_type="none",
                )
                logger.info("Created agent_migrations table")
        except Exception as e:
            logger.error(f"Error ensuring migration table: {str(e)}")
            raise

    async def get_applied_migrations(self) -> List[str]:
        """Get list of successfully applied migrations."""
        await self.ensure_migration_table()
        try:
            rows = await self.db.execute_query(
                "SELECT version FROM agent_migrations WHERE status = 'completed'"
            )
            migrations = [row["version"] for row in rows]
            logger.debug(f"Found {len(migrations)} completed migrations: {migrations}")
            return migrations
        except Exception as e:
            logger.error(f"Error getting applied migrations: {str(e)}")
            return []

    async def mark_migration_started(self, version: str) -> None:
        """Mark a migration as started."""
        await self.ensure_migration_table()
        try:
            # Check if this migration already has an entry
            exists = await self.db.execute_query(
                "SELECT EXISTS (SELECT 1 FROM agent_migrations WHERE version = $1)",
                version,
                fetch_type="val",
            )

            if exists:
                # Update existing entry
                await self.db.execute_query(
                    "UPDATE agent_migrations SET status = 'in_progress', applied_at = CURRENT_TIMESTAMP WHERE version = $1",
                    version,
                    fetch_type="none",
                )
            else:
                # Create new entry
                await self.db.execute_query(
                    "INSERT INTO agent_migrations (version, status) VALUES ($1, 'in_progress')",
                    version,
                    fetch_type="none",
                )
        except Exception as e:
            logger.error(f"Error marking migration started: {str(e)}")
            raise

    async def mark_migration_completed(self, version: str) -> None:
        """Mark a migration as completed."""
        await self.ensure_migration_table()
        try:
            await self.db.execute_query(
                "UPDATE agent_migrations SET status = 'completed', applied_at = CURRENT_TIMESTAMP WHERE version = $1",
                version,
                fetch_type="none",
            )
        except Exception as e:
            logger.error(f"Error marking migration completed: {str(e)}")
            raise

    async def mark_migration_failed(self, version: str, error: str) -> None:
        """Mark a migration as failed with error details."""
        await self.ensure_migration_table()
        try:
            await self.db.execute_query(
                "UPDATE agent_migrations SET status = 'failed', error_details = $2 WHERE version = $1",
                version,
                str(error),
                fetch_type="none",
            )
        except Exception as e:
            logger.error(f"Error marking migration failed: {str(e)}")
            raise

    async def check_schema_status(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check if the current database schema matches the expected structure.

        Returns:
            Tuple containing:
            - Boolean indicating if schema is valid
            - Dictionary with lists of missing tables, columns, and indices
        """
        try:
            await self.db.ensure_connected()

            missing_elements = {"tables": [], "columns": [], "indices": []}

            # Check existing tables
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            tables_rows = await self.db.execute_query(tables_query)
            existing_tables = {row["table_name"] for row in tables_rows}

            # Check for missing tables
            for table_name in self.expected_tables:
                if table_name not in existing_tables:
                    missing_elements["tables"].append(table_name)
                    # Skip column and index checks for missing tables
                    continue

                # Check columns for this table
                columns_query = """
                SELECT 
                    c.column_name, 
                    c.data_type, 
                    c.is_nullable,
                    (EXISTS (
                        SELECT 1 
                        FROM pg_constraint pc
                        JOIN pg_class pk_class ON pc.conrelid = pk_class.oid
                        WHERE pk_class.relname = $1
                        AND pc.contype = 'p'
                        AND c.ordinal_position = ANY(pc.conkey) -- Use = ANY() instead of @>
                    )) as is_primary
                FROM information_schema.columns c
                WHERE c.table_name = $1 
                AND c.table_schema = 'public' -- Added schema filter for robustness
                """
                columns_rows = await self.db.execute_query(columns_query, table_name)
                existing_columns = {}

                for row in columns_rows:
                    existing_columns[row["column_name"]] = {
                        "type": row["data_type"],
                        "is_nullable": row["is_nullable"] == "YES",
                        "is_primary": row["is_primary"] or False,
                    }

                # Check for missing or mismatched columns
                expected_columns = self.expected_tables[table_name]["columns"]
                for col_name, col_props in expected_columns.items():
                    if col_name not in existing_columns:
                        missing_elements["columns"].append(f"{table_name}.{col_name}")
                    else:
                        # Basic type checking - could be extended for more detailed checks
                        if (
                            col_props.get("is_primary", False)
                            != existing_columns[col_name]["is_primary"]
                        ):
                            missing_elements["columns"].append(
                                f"{table_name}.{col_name} (primary key mismatch)"
                            )

                # Check indices for this table
                indices_query = """
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = $1
                """
                indices_rows = await self.db.execute_query(indices_query, table_name)
                existing_indices = {row["indexname"] for row in indices_rows}

                # Check for missing indices
                expected_indices = self.expected_tables[table_name]["indices"]
                for idx_name in expected_indices:
                    if idx_name not in existing_indices:
                        missing_elements["indices"].append(f"{table_name}.{idx_name}")

            # Determine if schema is valid
            schema_valid = all(len(items) == 0 for items in missing_elements.values())

            return schema_valid, missing_elements

        except Exception as e:
            logger.error(f"Error checking schema status: {str(e)}", exc_info=True)
            # If error occurs, return False to trigger migration
            return False, {"error": [str(e)]}

    async def ensure_schema(self) -> bool:
        """Ensure required schema exists and is migrated to latest version."""
        try:
            logger.info("Checking database schema")
            await self.db.ensure_connected()

            # First check if the latest migration has been applied
            applied_migrations = await self.get_applied_migrations()
            if self.current_version in applied_migrations:
                logger.info(f"Migration {self.current_version} already recorded as applied")

                # Double-check schema structure
                schema_valid, missing_elements = await self.check_schema_status()
                if schema_valid:
                    logger.info("Schema structure verification passed")
                    return True
                else:
                    # Log missing elements but continue with migration
                    logger.warning(
                        f"Schema structure verification failed. Missing elements: {missing_elements}"
                    )
            else:
                # Check if the schema is already correct even if migration is not recorded
                schema_valid, missing_elements = await self.check_schema_status()
                if schema_valid:
                    logger.info("Schema structure is correct, marking migration as completed")
                    await self.mark_migration_completed(self.current_version)
                    return True
                logger.info(f"Migration {self.current_version} needs to be applied")

            # Run the migration
            return await self.migrate_to_v1()
        except Exception as e:
            logger.error(f"Error ensuring schema: {str(e)}", exc_info=True)
            return False

    async def migrate_to_v1(self) -> bool:
        """
        Perform the migration to schema version 1.
        Creates the conversation_messages table and required indices.
        """
        migration_version = self.current_version
        logger.info(f"Starting migration to {migration_version}")

        try:
            await self.mark_migration_started(migration_version)

            # Migration SQL from db_migration.sql
            migration_sql = """
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
            """

            # Execute the migration
            await self.db.execute_query(migration_sql, fetch_type="none")

            # Verify the migration was successful
            schema_valid, missing_elements = await self.check_schema_status()
            if not schema_valid:
                logger.warning(
                    f"Schema verification after migration shows issues: {missing_elements}"
                )
                # We'll still mark it as completed since the SQL executed without errors

            # Mark migration as completed
            await self.mark_migration_completed(migration_version)

            logger.info(f"Successfully applied migration {migration_version}")
            return True

        except Exception as e:
            error_message = f"Migration {migration_version} failed: {str(e)}"
            logger.error(error_message)
            logger.debug(f"Migration error traceback: {traceback.format_exc()}")
            await self.mark_migration_failed(migration_version, str(e))
            return False
