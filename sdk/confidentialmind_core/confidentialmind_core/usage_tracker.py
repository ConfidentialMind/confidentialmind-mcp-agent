import aiohttp
import aiofiles
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Union
from pydantic import BaseModel

from . import config
from .config_manager import ConfigManager

# Set up logging
logger = logging.getLogger(__name__)


def extract_api_key_middle_part(api_key: Optional[str]) -> Optional[str]:
    """
    Extract the middle part of a JWT API key for usage reporting.
    
    Args:
        api_key: API key to extract from
        
    Returns:
        Middle part of JWT token, or None if not a valid JWT
    """
    try:
        if not api_key or api_key.startswith("sk-"):
            return None
        
        # JWT tokens have 3 parts separated by dots
        parts = api_key.split('.')
        if len(parts) == 3:
            return parts[1]  # Return the middle part
        return None
    except Exception as e:
        logger.warning(f"Could not extract API key middle part: {e}")
        return None


class UsageType(str, Enum):
    """Types of usage that can be tracked"""
    INPUT_TOKENS = "inputTokens"
    OUTPUT_TOKENS = "outputTokens" 
    EMBEDDING_USAGE = "embeddingUsage"


class UsageRecord(BaseModel):
    """
    A single usage record matching the database structure.
    
    Attributes:
        originId: stackId of the service making the call
        targetId: stackId of the service being called  
        timestamp: Unix timestamp
        apiKey: middle part of JWT or null
        traceId: trace ID of the request
        type: type of usage (inputTokens, outputTokens, etc.)
        value: usage amount
    """
    originId: str
    targetId: str
    timestamp: int
    apiKey: Optional[str] = None
    traceId: str
    type: str
    value: int


class UsageReport(BaseModel):
    """Usage report format for the manager API"""
    usages: List[UsageRecord]


class DebugUsageTracker:
    """
    Minimalistic debug tracker for local development without manager API access.
    Just logs usage information to console/file for debugging purposes.
    """
    
    def __init__(self, log_to_file: bool = True):
        self.log_to_file = log_to_file
        if log_to_file:
            self.log_file = f"usage_debug_{datetime.now().strftime('%Y%m%d')}.log"
    
    async def _log_usage(self, message: str):
        """Log usage message to console and optionally to file"""
        print(f"[DEBUG USAGE] {message}")
        if self.log_to_file:
            try:
                async with aiofiles.open(self.log_file, "a", encoding="utf-8") as f:
                    await f.write(f"{datetime.now().isoformat()} - {message}\n")
            except Exception as e:
                print(f"[DEBUG USAGE] Failed to write to log file: {e}")
    
    async def report_usage(
        self, 
        usage_records: List[UsageRecord]
    ) -> bool:
        """Report usage records (debug implementation)"""
        await self._log_usage(f"Records: {len(usage_records)}")
        
        return True
    


class UsageTracker:
    """
    Production usage tracker that reports to the manager API.
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def _get_origin_stack_id(self) -> Optional[str]:
        """Get the stack ID of the current service (origin)"""
        try:
            # The service's own stack ID should be available from config manager
            return self.config_manager.service_id
        except Exception as e:
            print(f"Warning: Could not determine origin stack ID: {e}")
            return None
    
    def _get_manager_url(self) -> Optional[str]:
        """Get the manager API URL for usage reporting"""
        try:
            origin_stack_id = self._get_origin_stack_id()
            if not origin_stack_id:
                return None
            
            if config.LOCAL_DEV and hasattr(self.config_manager, '_ConfigManager__manager_url'):
                base_url = getattr(self.config_manager, '_ConfigManager__manager_url', None)
                if base_url:
                    # Remove existing /internal/{id} suffix and add /usage
                    base_url = base_url.replace(f'/internal/{origin_stack_id}', '')
                    return f"{base_url}/internal/{origin_stack_id}/usage"
            
            # Default production URL
            return f"{config.MANAGER_SERVICE_URL}/internal/{origin_stack_id}/usage"
        except Exception as e:
            print(f"Warning: Could not determine manager URL: {e}")
            return None
    
    async def report_usage(
        self, 
        usage_records: List[UsageRecord]
    ) -> bool:
        """
        Report usage records to the manager API.
        
        Args:
            usage_records: List of usage records to report
            
        Returns:
            bool: True if successful, False otherwise
        """
        manager_url = self._get_manager_url()
        if not manager_url:
            print("Warning: Could not determine manager URL for usage reporting")
            return False
        
        try:
            report = UsageReport(usages=usage_records)
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    manager_url,
                    json=report.model_dump(),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status >= 400:
                        response_text = await response.text()
                        print(f"Usage reporting failed with status {response.status}")
                        print(f"Response: {response_text}")
                        return False
            
            print(f"Successfully reported {len(usage_records)} usage records")
            return True
            
        except Exception as e:
            print(f"Error reporting usage: {e}")
            return False
    


def get_usage_tracker() -> Union[UsageTracker, DebugUsageTracker]:
    """
    Factory function to choose the appropriate usage tracker implementation.
    
    Returns:
        DebugUsageTracker for local development,
        UsageTracker for production scenarios.
    """
    if config.LOCAL_DEV and config.LOCAL_CONFIGS:
        return DebugUsageTracker()
    return UsageTracker()