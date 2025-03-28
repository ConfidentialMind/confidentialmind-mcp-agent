from typing import Dict, Optional, Tuple

from confidentialmind_core.config_manager import (
    ArrayConnectorSchema,
    ConfigManager,
    ConnectorSchema,
)
from pydantic import BaseModel

from src.connectors.llm import LLMConnector
from src.mcp.registry import MCPRegistry


class ConfidentialMindConnector:
    """Integration with ConfidentialMind SDK for LLM and MCP connectors"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.llm_config_id = "agent_llm"
        self.mcp_config_ids = {}
        self.connector_schemas = []
        self.array_connector_schemas = []

    def init_connector(self, config_model: Optional[BaseModel] = None):
        """Initialize connection with ConfidentialMind stack"""
        # Define LLM connector
        self.array_connector_schemas = [
            ArrayConnectorSchema(
                type="llm",
                label="Select Agent LLM",
                config_id=self.llm_config_id,
                description="Choose the LLM model that will power the agent",
            )
        ]

        # Add MCP connectors
        for mcp_id, mcp_info in self.mcp_config_ids.items():
            self.connector_schemas.append(
                ConnectorSchema(
                    type=mcp_info.get("type", "api"),
                    label=mcp_info.get("label", f"MCP: {mcp_id}"),
                    config_id=mcp_id,
                    description=mcp_info.get(
                        "description", f"Connection for {mcp_id} services"
                    ),
                )
            )

        # Initialize config manager
        self.config_manager.init_manager(
            config_model=config_model,
            connectors=self.connector_schemas,
            array_connectors=self.array_connector_schemas,
        )

    def register_mcp_connector(
        self, mcp_id: str, mcp_type: str, label: str, description: str = ""
    ):
        """Register an MCP connector"""
        self.mcp_config_ids[mcp_id] = {
            "type": mcp_type,
            "label": label,
            "description": description,
        }

        # If config manager is already initialized, update connectors
        if (
            hasattr(self.config_manager, "connectors")
            and self.config_manager.connectors
        ):
            self.connector_schemas.append(
                ConnectorSchema(
                    type=mcp_type,
                    label=label,
                    config_id=mcp_id,
                    description=description,
                )
            )
            self.config_manager.update_connectors(self.connector_schemas)

    def get_api_parameters(self, config_id: str) -> Tuple[str, Dict[str, str]]:
        """Get API parameters for a connector"""
        from confidentialmind_core.config_manager import get_api_parameters

        return get_api_parameters(config_id)

    def get_llm_connector(self) -> LLMConnector:
        """Get an LLM connector configured through ConfidentialMind"""
        urls_or_url = self.config_manager.getUrlForConnector(self.llm_config_id)

        if isinstance(urls_or_url, list):
            llm_urls = urls_or_url
            llm_headers = []

            for i, url in enumerate(llm_urls):
                _, headers = self.get_api_parameters(f"{self.llm_config_id}_{i}")
                llm_headers.append(headers)

            return LLMConnector(base_urls=llm_urls, headers_list=llm_headers)

        else:
            base_url, headers = self.get_api_parameters(self.llm_config_id)
            return LLMConnector(base_url=base_url, headers=headers)

    def get_mcp_registry(self) -> MCPRegistry:
        """Create and configure MCP registry from ConfidentialMind connectors"""
        from src.agent_orchestrator.mcp.examples.database import DatabaseMCP
        from src.agent_orchestrator.mcp.examples.monitoring import MonitoringMCP

        registry = MCPRegistry()

        # Create and register MCPs based on configured connectors
        for mcp_id, mcp_info in self.mcp_config_ids.items():
            try:
                base_url, headers = self.get_api_parameters(mcp_id)

                # Instantiate appropriate MCP based on type
                mcp_type = mcp_info.get("type", "").lower()

                if "database" in mcp_type or "db" in mcp_type:
                    mcp = DatabaseMCP(base_url=base_url, headers=headers)
                elif "monitor" in mcp_type:
                    mcp = MonitoringMCP(base_url=base_url, headers=headers)
                elif "postgres" in mcp_type and "service_type" in mcp_info:
                    # Import dynamically to avoid circular imports
                    from examples.custom_mcp.postgres_service_mcp import PostgreSQLMCP

                    mcp = PostgreSQLMCP(connection_string=base_url)
                # Add more MCP types as needed
                else:
                    # Generic API-based MCP
                    continue  # Skip for now

                registry.register(mcp)
            except Exception as e:
                print(f"Error initializing MCP {mcp_id}: {str(e)}")

        return registry

    def register_postgres_service(self, service_type: str, label: str = None):
        """Register a PostgreSQL service as an MCP"""
        if not label:
            label = f"PostgreSQL {service_type.capitalize()}"

        mcp_id = f"postgres_{service_type}"
        self.register_mcp_connector(
            mcp_id=mcp_id,
            mcp_type="postgres_service",
            label=label,
            description=f"Connection to PostgreSQL {service_type} service",
        )

        # Store service type for MCP creation
        if mcp_id in self.mcp_config_ids:
            self.mcp_config_ids[mcp_id]["service_type"] = service_type
