"""Database operations for the PostgreSQL MCP server."""

from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from asyncpg import Pool

from src.mcp.postgres.settings import PostgresSettings
from src.mcp.postgres.utils import sanitize_error
from src.mcp.postgres.validators import validate_query


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, settings: PostgresSettings):
        """Initialize the database manager.
        
        Args:
            settings: The PostgreSQL settings
        """
        self.settings = settings
        self.pool: Optional[Pool] = None
    
    async def get_connection_pool(self) -> Pool:
        """Get or create the database connection pool.
        
        Returns:
            Pool: The database connection pool
        """
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.settings.host,
                port=self.settings.port,
                user=self.settings.user,
                password=self.settings.password,
                database=self.settings.database,
                timeout=self.settings.connection_timeout,
                max_size=self.settings.max_connections,
            )
        return self.pool
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def execute_query(self, query: str) -> Tuple[bool, Any]:
        """Execute a read-only SQL query.
        
        Args:
            query: The SQL query to execute
            
        Returns:
            Tuple[bool, Any]: (success, results/error_message)
        """
        # Validate the query
        is_valid, reason = validate_query(query)
        if not is_valid:
            return False, reason
        
        try:
            pool = await self.get_connection_pool()
            async with pool.acquire() as conn:
                # Set statement timeout
                await conn.execute(f"SET statement_timeout TO {self.settings.statement_timeout}")
                
                # Execute the query
                results = await conn.fetch(query)
                
                # Convert results to list of dicts
                return True, [dict(row) for row in results]
        except Exception as e:
            return False, sanitize_error(str(e))
    
    async def get_all_tables(self) -> List[Dict[str, str]]:
        """Get a list of all tables in the database.
        
        Returns:
            List[Dict[str, str]]: List of tables with schema and table name
        """
        pool = await self.get_connection_pool()
        async with pool.acquire() as conn:
            # Set a timeout for the query
            await conn.execute(f"SET statement_timeout TO {self.settings.statement_timeout}")
            tables = await conn.fetch("""
                SELECT table_name, table_schema 
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
            """)
            return [dict(table) for table in tables]
    
    async def get_table_schema(self, schema_name: str, table_name: str) -> Dict[str, Any]:
        """Get the schema for a specific table.
        
        Args:
            schema_name: The name of the schema
            table_name: The name of the table
            
        Returns:
            Dict[str, Any]: Table schema information with columns and primary keys
        """
        pool = await self.get_connection_pool()
        async with pool.acquire() as conn:
            # Set a timeout for the query
            await conn.execute(f"SET statement_timeout TO {self.settings.statement_timeout}")
            
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, 
                       column_default, character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position
            """, schema_name, table_name)
            
            primary_keys = await conn.fetch("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = ($1 || '.' || $2)::regclass AND i.indisprimary
            """, schema_name, table_name)
            
            pk_list = [pk['attname'] for pk in primary_keys]
            
            return {
                "table_name": table_name,
                "schema_name": schema_name,
                "columns": [dict(col) for col in columns],
                "primary_keys": pk_list
            }
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get general information about the database.
        
        Returns:
            Dict[str, Any]: Database information including name, schemas, and table counts
        """
        pool = await self.get_connection_pool()
        async with pool.acquire() as conn:
            # Set statement timeout
            await conn.execute(f"SET statement_timeout TO {self.settings.statement_timeout}")
            
            # Get database name
            db_name = await conn.fetchval("SELECT current_database()")
            
            # Get schema information
            schemas = await conn.fetch("""
                SELECT schema_name, COUNT(table_name) as table_count
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                GROUP BY schema_name
                ORDER BY schema_name
            """)
            
            # Get total tables
            total_tables = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            """)
            
            return {
                "database_name": db_name,
                "total_tables": total_tables,
                "schemas": [dict(schema) for schema in schemas]
            }