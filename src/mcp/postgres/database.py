"""Database operations for the PostgreSQL MCP server."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
from asyncpg import Pool

from .settings import PostgresSettings
from .utils import sanitize_error
from .validators import validate_query

# Initialize logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, settings: PostgresSettings):
        """Initialize the database manager.
        
        Args:
            settings: The PostgreSQL settings
        """
        self.settings = settings
        self.pool: Optional[Pool] = None
        self.log_interval_count = 0  # Counter for periodic logging
    
    async def get_connection_pool(self) -> Pool:
        """Get or create the database connection pool.
        
        Logs pool statistics periodically to help monitor connection usage.
        
        Returns:
            Pool: The database connection pool
        """
        if self.pool is None:
            logger.info(
                f"Creating new connection pool with max_size={self.settings.max_connections}, "
                f"timeout={self.settings.connection_timeout}s"
            )
            self.pool = await asyncpg.create_pool(
                host=self.settings.host,
                port=self.settings.port,
                user=self.settings.user,
                password=self.settings.password,
                database=self.settings.database,
                timeout=self.settings.connection_timeout,
                max_size=self.settings.max_connections,
                min_size=2,  # Keep minimum 2 connections ready
            )
        
        # Log pool statistics every 10 calls to help with monitoring
        self.log_interval_count = (self.log_interval_count + 1) % 10
        if self.log_interval_count == 0 and self.pool:
            free_connections = self.pool.get_size() - self.pool.get_idle_size()
            used_percent = (free_connections / self.pool.get_size()) * 100 if self.pool.get_size() > 0 else 0
            logger.info(
                f"Pool status: size={self.pool.get_size()}, "
                f"free={free_connections}, "
                f"usage={used_percent:.1f}%"
            )
            
            # Log warning if the pool is getting close to max capacity
            if used_percent > 80:
                logger.warning(
                    f"Connection pool is at {used_percent:.1f}% capacity. "
                    f"Consider increasing max_connections setting if this happens frequently."
                )
                
        return self.pool
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            logger.info("Closing database connection pool")
            await self.pool.close()
            self.pool = None
    
    async def execute_query(self, query: str) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """Execute a read-only SQL query.
        
        Args:
            query: The SQL query to execute
            
        Returns:
            Tuple[bool, Union[List[Dict[str, Any]], str]]: (success, results/error_message)
                If success is True, the second element is a list of result rows as dictionaries.
                If success is False, the second element is an error message string.
        """
        query_preview = query[:50] + ('...' if len(query) > 50 else '')
        logger.debug(f"Validating query: {query_preview}")
        
        # Validate the query
        is_valid, reason = validate_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {reason} - Query: {query_preview}")
            return False, reason
        
        try:
            pool = await self.get_connection_pool()
            async with pool.acquire() as conn:
                # Set statement timeout
                logger.debug(f"Setting statement timeout to {self.settings.statement_timeout}ms")
                await conn.execute(f"SET statement_timeout TO {self.settings.statement_timeout}")
                
                # Execute the query
                logger.info(f"Executing query: {query_preview}")
                start_time = time.time()
                results = await conn.fetch(query)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Log execution statistics
                row_count = len(results)
                logger.info(f"Query completed in {execution_time:.2f}ms, returned {row_count} rows")
                
                # Convert results to list of dicts
                return True, [dict(row) for row in results]
        except asyncio.TimeoutError:
            logger.error(f"Query timeout after {self.settings.statement_timeout}ms: {query_preview}")
            return False, f"Query timed out after {self.settings.statement_timeout/1000:.1f} seconds"
        except Exception as e:
            error_msg = sanitize_error(str(e))
            logger.error(f"Query error: {error_msg} - Query: {query_preview}")
            return False, error_msg
    
    async def get_all_tables(self) -> List[Dict[str, str]]:
        """Get a list of all tables in the database.
        
        Returns:
            List[Dict[str, str]]: List of tables with schema and table name.
                Each dict contains 'table_name' and 'table_schema' keys.
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
    
    async def get_table_schema(self, schema_name: str, table_name: str) -> Dict[str, Union[str, List[Dict[str, Any]], List[str]]]:
        """Get the schema for a specific table.
        
        Args:
            schema_name: The name of the schema
            table_name: The name of the table
            
        Returns:
            Dict[str, Union[str, List[Dict[str, Any]], List[str]]]: Table schema information with the following keys:
                - table_name: str - The name of the table
                - schema_name: str - The name of the schema
                - columns: List[Dict[str, Any]] - List of column definitions
                - primary_keys: List[str] - List of primary key column names
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
    
    async def get_database_info(self) -> Dict[str, Union[str, int, List[Dict[str, Union[str, int]]]]]:
        """Get general information about the database.
        
        Returns:
            Dict[str, Union[str, int, List[Dict[str, Union[str, int]]]]]: Database information with the following keys:
                - database_name: str - The name of the database
                - total_tables: int - The total number of tables
                - schemas: List[Dict[str, Union[str, int]]] - List of schemas with table counts
                  Each schema dict contains 'schema_name' (str) and 'table_count' (int)
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